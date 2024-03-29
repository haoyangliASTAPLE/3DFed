import argparse
import shutil
from datetime import datetime
import yaml
from copy import deepcopy
from prompt_toolkit import prompt
from tqdm import tqdm
from helper import Helper
from utils.utils import *
logger = logging.getLogger('logger')

def train(hlpr: Helper, epoch, model, optimizer, train_loader, attack=True, global_model=None):
    criterion = hlpr.task.criterion
    model.train()
    for i, data in tqdm(enumerate(train_loader)):
        batch = hlpr.task.get_batch(i, data)
        model.zero_grad()
        loss = hlpr.attack.compute_blind_loss(model, criterion, batch, attack, global_model)
        loss.backward()
        optimizer.step()

        if i == hlpr.params.max_batch_id:
            break
    return

def test(hlpr: Helper, epoch, backdoor=False, model=None):
    if model is None:
        model = hlpr.task.model
    model.eval()
    hlpr.task.reset_metrics()
    with torch.no_grad():
        for i, data in tqdm(enumerate(hlpr.task.test_loader)):
            batch = hlpr.task.get_batch(i, data)
            if backdoor:
                batch = hlpr.attack.synthesizer.make_backdoor_batch(batch,
                                                                    test=True,
                                                                    attack=True)

            outputs = model(batch.inputs)
            hlpr.task.accumulate_metrics(outputs=outputs, labels=batch.labels)
    metric = hlpr.task.report_metrics(epoch,
                             prefix=f'Backdoor {str(backdoor):5s}. Epoch: ')
    return metric

def run_fl_round(hlpr: Helper, epoch):
    global_model = hlpr.task.model
    local_model = hlpr.task.local_model
    round_participants = hlpr.task.sample_users_for_round(epoch)
    weight_accumulator = hlpr.task.get_empty_accumulator()

    for user in tqdm(round_participants):
        hlpr.task.copy_params(global_model, local_model)
        optimizer = hlpr.task.make_optimizer(local_model)
        if user.compromised:
            if not user.user_id == 0:
                continue
            for local_epoch in tqdm(range(hlpr.params.fl_poison_epochs)):
                train(hlpr, local_epoch, local_model, optimizer,
                        user.train_loader, attack=True, global_model=global_model)
        else:
            for local_epoch in range(hlpr.params.fl_local_epochs):
                train(hlpr, local_epoch, local_model, optimizer,
                        user.train_loader, attack=False)
        local_update = hlpr.attack.get_fl_update(local_model, global_model)
        hlpr.save_update(model=local_update, userID=user.user_id)
        if user.compromised:
            hlpr.attack.local_dataset = deepcopy(user.train_loader)

    hlpr.attack.perform_attack(global_model, epoch)
    hlpr.defense.aggr(weight_accumulator, global_model)
    hlpr.task.update_global_model(weight_accumulator, global_model)

def run(hlpr: Helper):
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        run_fl_round(hlpr, epoch)
        metric = test(hlpr, epoch, backdoor=False)
        hlpr.record_accuracy(metric, test(hlpr, epoch, backdoor=True), epoch)

        hlpr.save_model(hlpr.task.model, epoch, metric)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backdoors')
    parser.add_argument('--params', dest='params', required=True)
    parser.add_argument('--name', dest='name', required=True)
    args = parser.parse_args()
    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)
    params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
    params['name'] = args.name

    helper = Helper(params)
    logger.warning(create_table(params))

    try:
        run(helper)
    except (KeyboardInterrupt):
        if helper.params.log:
            answer = prompt('\nDelete the repo? (y/n): ')
            if answer in ['Y', 'y', 'yes']:
                logger.error(f"Fine. Deleted: {helper.params.folder_path}")
                shutil.rmtree(helper.params.folder_path)
            else:
                logger.error(f"Aborted training. "
                             f"Results: {helper.params.folder_path}. ")
        else:
            logger.error(f"Aborted training. No output generated.")
    helper.remove_update()