import os.path
import time
import math

import torch.optim.lr_scheduler

from train import train
from evaluate import evaluate


def learn(train_loader, val_loader, model, optimizer, loss_function, num_epochs, device, classes,
          task, tb_writer, params, mining_function, class_loss_function=None, class_mining_function=None):

    step_size = params['step_size']
    max_norm = params['max_norm']
    lr_lim = params['learning_rate_lim']

    if not params['should_learn']:
        print(f"| Did not learn, as 'should_learn' was false")
        return model, None
    else:
        # Print runtime information
        if 'num_runs' in params and 'num_repeats' in params:
            run_formatter = int(math.log10(params['num_runs'])) + 1
            rep_formatter = int(math.log10(params['num_repeats'])) + 1

            print(f"| Learning phase"
                  f" - Run {params['run_idx'] + 1:{run_formatter}.0f}/{params['num_runs']}"
                  f" - Rep {params['rep_idx'] + 1:{rep_formatter}.0f}/{params['num_repeats']}")
        else:
            print('-' * 28, 'Learning phase', '-' * 28)

    learn_start_time = time.time()

    prev_val_acc = None

    epoch_formatter = int(math.log10(num_epochs)) + 1

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=.1)
    curr_lr = optimizer.param_groups[0]['lr']
    prev_lr = curr_lr

    bad_val_counter = 0
    bad_val_lim = params['bad_val_lim_first']

    learn_info = dict()

    best_epoch_acc_state = {'accuracy': -1}
    state = dict()

    for epoch_idx in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        epoch_status_message = f"| Epoch {epoch_idx:{epoch_formatter}.0f}/{num_epochs} "

        if lr_lim is not None:
            if curr_lr <= lr_lim:
                if bad_val_lim-bad_val_counter <= 0:
                    break
                print(epoch_status_message + f"| LR: {curr_lr:g} "
                                             f"| Breaking in {bad_val_lim-bad_val_counter} more un-increasing vals |")
            elif prev_lr == curr_lr:
                print(epoch_status_message + f"| LR: {curr_lr:g} "
                                             f"| Step after {bad_val_lim-bad_val_counter} more un-increasing vals |")
            else:
                print(epoch_status_message + f"| Stepped the LR to: {curr_lr:g}")

        prev_lr = curr_lr

        # Train the epoch
        model, train_info = train(
            data_loader=train_loader,
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            class_loss_function=class_loss_function,
            mining_function=mining_function,
            class_mining_function=class_mining_function,
            device=device,
            task=task,
            epoch_idx=epoch_idx,
            num_epochs=num_epochs,
            classes=classes,
            max_norm=max_norm,
            tb_writer=tb_writer,
            params=params
        )

        # Validate the epoch
        val_info, val_message = evaluate(
            train_loader=train_loader,
            eval_loader=val_loader,
            model=model,
            task=task,
            device=device,
            is_test=False,
            embedding_dims=params['embedding_dims']
        )

        epoch_status_message += val_message
        epoch_time = time.time() - epoch_start_time

        print(epoch_status_message + f"| Duration: {epoch_time:.2f}s | ------------")

        # Save run info
        learn_info[epoch_idx] = {
            'train_info': train_info,
            'val_info': val_info,
            'time': epoch_time,
            'learning_rate': f'{curr_lr:g}',
        }

        # Store the best epoch based on validation accuracy
        #  which is regular accuracy for classification and MAP@R for metric
        if val_info['accuracy'] > best_epoch_acc_state['accuracy']:
            best_epoch_acc_state = {
                'epoch': epoch_idx,
                'net': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'accuracy': val_info['accuracy']
            }

        # Save a checkpoint when the epoch finishes
        state = {'epoch': epoch_idx, 'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
        file_path = f"./saves/checkpoints/last_net_run/checkpoint_{epoch_idx}.pth"
        torch.save(state, file_path)

        if lr_lim is not None:
            if prev_val_acc is not None and prev_val_acc >= val_info['accuracy']:
                bad_val_counter += 1

                if bad_val_counter >= bad_val_lim and curr_lr > lr_lim:
                    # Step the learning rate
                    scheduler.step()
                    curr_lr = optimizer.param_groups[0]['lr']
                    bad_val_counter = 0
                    bad_val_lim = params['bad_val_lim']
                    if params['load_best_post_lr_step']:
                        print(f"| Stepping and loading the best model "
                              f"| Epoch: {best_epoch_acc_state['epoch']} "
                              f"| Accuracy: {best_epoch_acc_state['accuracy']:.6f}")
                        model.load_state_dict(best_epoch_acc_state['net'])
                        #optimizer.load_state_dict(best_epoch_acc_state['optimizer'])
            else:
                prev_val_acc = val_info['accuracy']
        else:
            scheduler.step()

    print(f'| Finished learning | Learning time: {(time.time()-learn_start_time):.2f}s')

    file_path_final = "./saves/checkpoints/final/"
    file_name_last = "{}_ṛun{}_rep{}_e{}.pth".format(
        params['run_name'].split('.')[0], params['run_idx']+1, params['rep_idx']+1, state['epoch']
    )
    file_name_best = "{}_ṛun{}_rep{}_e{}_best.pth".format(
        params['run_name'].split('.')[0], params['run_idx']+1, params['rep_idx']+1, best_epoch_acc_state['epoch']
    )

    file_path_last = os.path.join(file_path_final, file_name_last)
    file_path_best = os.path.join(file_path_final, file_name_best)

    torch.save(state, file_path_last)
    torch.save(best_epoch_acc_state, file_path_best)

    return model, learn_info
