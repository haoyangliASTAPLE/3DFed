import argparse
import shutil
from datetime import datetime

import yaml
from prompt_toolkit import prompt
from tqdm import tqdm

# noinspection PyUnresolvedReferences
from dataset.pipa import Annotations  # legacy to correctly load dataset.
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

        hlpr.report_training_losses_scales(i, epoch)
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
                             prefix=f'Backdoor {str(backdoor):5s}. Epoch: ',
                             tb_writer=hlpr.tb_writer,
                             tb_prefix=f'Test_backdoor_{str(backdoor):5s}')

    return metric


def run(hlpr: Helper):
    acc = test(hlpr, 0, backdoor=False)
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        train(hlpr, epoch, hlpr.task.model, hlpr.task.optimizer,
              hlpr.task.train_loader)
        acc = test(hlpr, epoch, backdoor=False)
        test(hlpr, epoch, backdoor=True)
        hlpr.save_model(hlpr.task.model, epoch, acc)


def fl_run(hlpr: Helper):
    for epoch in range(hlpr.params.start_epoch,
                       hlpr.params.epochs + 1):
        run_fl_round(hlpr, epoch)
        metric = test(hlpr, epoch, backdoor=False)
        hlpr.record_accuracy(metric, test(hlpr, epoch, backdoor=True), epoch)

        hlpr.save_model(hlpr.task.model, epoch, metric)


def run_fl_round(hlpr: Helper, epoch):
    global_model = hlpr.task.model
    local_model = hlpr.task.local_model

    round_participants = hlpr.task.sample_users_for_round(epoch)
    weight_accumulator = hlpr.task.get_empty_accumulator()
    adv_dataset = None

    for user in tqdm(round_participants):
        hlpr.task.copy_params(global_model, local_model)
        optimizer = hlpr.task.make_optimizer(local_model)
        
        if user.compromised:
            # if hlpr.params.fl_camouflage and not user.user_id == 0:
            if not user.user_id == 0:
                continue
            for local_epoch in tqdm(range(hlpr.params.fl_poison_epochs)):
                train(hlpr, local_epoch, local_model, optimizer,
                        user.train_loader, attack=True, global_model=global_model)
            if hlpr.params.fl_camouflage:
                print("Camouflage: Finish training backdoor model")
                logger.warning("Camouflage: eu distance of backdoor model "+str(user.user_id)+" : "+
                    str(hlpr.task.calculate_eu_dist(local_model, global_model)))
                logger.warning("Camouflage: cos sim of backdoor model "+str(user.user_id)+" : "+
                    str(hlpr.task.calculate_cos_sim(local_model, global_model)[0][1]))
            test(hlpr, epoch=999, backdoor=False, model=local_model)
            test(hlpr, epoch=999, backdoor=True, model=local_model)
        else:
            for local_epoch in range(hlpr.params.fl_local_epochs):
                train(hlpr, local_epoch, local_model, optimizer,
                        user.train_loader, attack=False)
        local_update = hlpr.task.get_fl_update(local_model, global_model)
        if user.compromised and not hlpr.params.fl_camouflage:
            hlpr.attack.fl_scale_update(local_update)
        hlpr.task.save_update(model=local_update, userID=user.user_id)
        if user.compromised:
            adv_dataset = user.train_loader
    hlpr.task.camouflage(adv_dataset, hlpr, epoch)
    hlpr.task.robust_defense(weight_accumulator, epoch)

    hlpr.task.update_global_model(weight_accumulator, global_model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Backdoors')
    parser.add_argument('--params', dest='params', default='utils/params.yaml')
    parser.add_argument('--name', dest='name', required=True)
    parser.add_argument('--commit', dest='commit',
                        default=get_current_git_hash())

    args = parser.parse_args()

    with open(args.params) as f:
        params = yaml.load(f, Loader=yaml.FullLoader)

    params['current_time'] = datetime.now().strftime('%b.%d_%H.%M.%S')
    params['commit'] = args.commit
    params['name'] = args.name

    helper = Helper(params)
    # logger.warning(create_table(params))

    try:
        if helper.params.fl:
            fl_run(helper)
            helper.task.remove_update()
        else:
            run(helper)
    except (KeyboardInterrupt):
        if helper.params.fl:
            helper.task.remove_update()
        if helper.params.log:
            answer = prompt('\nDelete the repo? (y/n): ')
            if answer in ['Y', 'y', 'yes']:
                logger.error(f"Fine. Deleted: {helper.params.folder_path}")
                shutil.rmtree(helper.params.folder_path)
                if helper.params.tb:
                    shutil.rmtree(f'runs/{args.name}')
            else:
                logger.error(f"Aborted training. "
                             f"Results: {helper.params.folder_path}. "
                             f"TB graph: {args.name}")
        else:
            logger.error(f"Aborted training. No output generated.")
