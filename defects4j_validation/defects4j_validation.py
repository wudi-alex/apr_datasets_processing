import shutil
import os
import time
import defects4j_command
from datasets import load_from_disk, Dataset

DEFECTS4J_DIR = os.path.abspath(__file__)[: os.path.abspath(__file__).rindex('/') + 1]


def insert_fix(filename, start_line, end_line, patch):
    """
    end_row is included in the buggy lines / buggy function
    """
    with open(filename, 'r') as file:
        data = file.readlines()

    with open(filename, 'w') as file:
        for i in range(start_line - 1):
            file.write(data[i])
        file.write(patch.strip() + '\n')
        for i in range(end_line, len(data)):
            file.write(data[i])


def validate_defects4j(input):
    plausible, total = 0, 0

    proj, bug_id, start_line, end_line, path, patch = parse_input(input)

    # tmp_dir = f'/Users/alex.wu/defects4j_projects_buggy/{proj}/{proj}_{bug_id}/'
    tmp_dir = f'/tmp/tmp_benchmarks/defects4j/{proj}_{bug_id}/'

    if not os.path.exists(tmp_dir):
        defects4j_command.command_with_timeout(['mkdir', tmp_dir])

    output = []

    defects4j_command.clean_tmp_folder(tmp_dir)
    defects4j_command.checkout_defects4j_project(proj, bug_id + 'b', tmp_dir)

    # "Mockito needs separate compilation"
    if proj == "Mockito":
        defects4j_command.compile_fix(tmp_dir)

    # check standard test time
    start_time = time.time()
    init_out, init_err = defects4j_command.defects4j_test_suite(tmp_dir)
    standard_time = int(time.time() - start_time)

    # check failed test cases
    failed_test_cases = str(init_out).split(' - ')[1:]
    for i, failed_test_case in enumerate(failed_test_cases):
        failed_test_cases[i] = failed_test_case.strip()
    init_fail_num = len(failed_test_cases)
    print(init_fail_num, str(standard_time) + 's')

    # check triggering failed test cases
    trigger, err = defects4j_command.defects4j_trigger(tmp_dir)
    triggers = trigger.strip().split('\n')
    for i, trigger in enumerate(triggers):
        triggers[i] = trigger.strip()
    print('trigger number:', len(triggers))

    current_is_correct = False
    for rank, patch in enumerate(list(set(patch))):
        filename = tmp_dir + path
        shutil.copyfile(filename, filename + '.bak')

        insert_fix(filename, int(start_line), int(end_line), patch.strip())

        if proj == 'Mockito':
            # Mockito needs seperate compile
            defects4j_command.compile_fix(tmp_dir)

        # trigger cases is few and total time is long, we test trigger cases first.
        outs = []
        correctness = None
        out, err = '', ''
        start_time = time.time()
        if standard_time >= 10 and len(triggers) <= 5:
            for trigger in triggers:
                out, err = defects4j_command.defects4j_test_one(tmp_dir, trigger,
                                                                timeout=min(200, int(1.5 * standard_time)))
                if 'TIMEOUT' in str(err) or 'TIMEOUT' in str(out):
                    print(plausible, total, rank, 'Time out for patch: ', patch,
                          str(int(time.time() - start_time)) + 's')
                    correctness = 'timeout'
                    break
                elif 'FAIL' in str(err) or 'FAIL' in str(out):
                    print(plausible, total, rank, 'Uncompilable patch:', patch,
                          str(int(time.time() - start_time)) + 's')
                    correctness = 'uncompilable'
                    break
                elif "Failing tests: 0" in str(out):
                    continue
                else:
                    outs += str(out).split(' - ')[1:]
        if len(set(outs)) >= len(triggers):
            # does not pass any one more
            print(plausible, total, rank, 'Wrong patch:', patch,
                  str(int(time.time() - start_time)) + 's')
            correctness = 'wrong'

        if correctness is None:
            # pass at least one more trigger case
            # have to pass all non-trigger
            out, err = defects4j_command.defects4j_test_suite(tmp_dir, timeout=min(200, int(1.5 * standard_time)))

            if 'TIMEOUT' in str(err) or 'TIMEOUT' in str(out):
                print(plausible, total, rank, 'Time out for patch: ', patch,
                      str(int(time.time() - start_time)) + 's')
                correctness = 'timeout'
            elif 'FAIL' in str(err) or 'FAIL' in str(out):
                print(plausible, total, rank, 'Uncompilable patch:', patch,
                      str(int(time.time() - start_time)) + 's')
                correctness = 'uncompilable'
            elif "Failing tests: 0" in str(out):
                if not current_is_correct:
                    current_is_correct = True
                    plausible += 1
                print(plausible, total, rank, 'Plausible patch:', patch,
                      str(int(time.time() - start_time)) + 's')
                correctness = 'plausible'
            elif len(str(out).split(' - ')[1:]) < init_fail_num:
                # fail less, could be correct
                current_failed_test_cases = str(out).split(' - ')[1:]
                no_new_fail = True
                for current_failed_test_case in current_failed_test_cases:
                    if current_failed_test_case.strip() not in failed_test_cases:
                        no_new_fail = False
                        break
                if no_new_fail:
                    # fail less and no new fail cases, could be plausible
                    if not current_is_correct:
                        current_is_correct = True
                        plausible += 1
                    print(plausible, total, rank, 'Plausible patch:', patch,
                          str(int(time.time() - start_time)) + 's')
                    correctness = 'plausible'
                else:
                    print(plausible, total, rank, 'Wrong patch:', patch,
                          str(int(time.time() - start_time)) + 's')
                    correctness = 'wrong'
            else:
                print(plausible, total, rank, 'Wrong patch:', patch,
                      str(int(time.time() - start_time)) + 's')
                correctness = 'wrong'

        output.append({'patch': patch, 'correctness': correctness, 'test_message': out})
        shutil.copyfile(filename + '.bak', filename)

    input['test_res'] = output
    return input


def wrap_validate_func(batch):
    try:
        return validate_defects4j(batch)
    except Exception as e:
        print(f"Error: {e}")
        return None


def parse_input(input):
    buggy_info = input['buggyInfo']
    proj, bug_id, start_line, end_line, path = buggy_info['projectName'].split('_')[0], \
        buggy_info['projectName'].split('_')[1], buggy_info['startLine'], buggy_info['endLine'], buggy_info[
        'filePath'].replace(f"/Users/alex.wu/defects4j_projects_buggy/{buggy_info['projectName']}", '')
    end_line = str(int(end_line) - 1)
    patch = [
        i.replace(input['input'], '').replace('<s>', '').replace('</s>', '').replace('<unk>', '').replace('<EOT>', '')
        for i in input['gen']]
    return (
        proj.capitalize().replace('Jacksondatabind', 'JacksonDatabind').replace('Jacksoncore', 'JacksonCore').replace(
            'Jacksonxml', 'JacksonXml').replace('Jxpath', 'JxPath'),
        bug_id, start_line, end_line, path, patch)


DATASET_NAME = 'defects4j_vanilla_gen_10'


def gen_validate_dataset(dataset):
    print(f"start validating...")
    updated_dataset = dataset.map(wrap_validate_func, num_proc=16)
    updated_dataset.save_to_disk(f"dataset_validated/{DATASET_NAME}_validation")
    print(f"finish validating...")


dataset = load_from_disk(f'dataset_to_be_validated/{DATASET_NAME}')
gen_validate_dataset(dataset)
