import time
import math

import torch.optim.lr_scheduler

from train import train
from evaluate import evaluate, evaluate_metric


def learn(train_loader, val_loader, model, optimizer, loss_function, num_epochs, device, classes, lr_lim, loss_type,
          task, max_norm, step_size, tb_writer):

    learn_start_time = time.time()

    prev_val_acc = None

    epoch_formatter = int(math.log10(num_epochs)) + 1

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=.1)
    curr_lr = optimizer.param_groups[0]['lr']
    prev_lr = curr_lr

    bad_val_counter = 0
    bad_val_lim = 30

    learn_info = dict()

    for epoch_idx in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        epoch_status_message = f"| Epoch {epoch_idx:{epoch_formatter}.0f}/{num_epochs} "

        if lr_lim is not None:
            if curr_lr <= lr_lim:
                if bad_val_lim-bad_val_counter <= 0:
                    break
                print(epoch_status_message + f"| Breaking in {bad_val_lim-bad_val_counter} more un-increasing vals |")
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
            device=device,
            loss_type=loss_type,
            task=task,
            epoch_idx=epoch_idx,
            num_epochs=num_epochs,
            classes=classes,
            max_norm=max_norm,
            tb_writer=tb_writer
        )

        # Validate the epoch
        if task == 'classification':
            val_info = evaluate(
                data_loader=val_loader,
                model=model,
                device=device,
                is_test=False
            )

            epoch_status_message += f"| Val accuracy: {val_info['accuracy']:.6f} "

        elif task == 'metric':
            val_info = evaluate_metric(
                train_loader=train_loader,
                eval_loader=val_loader,
                model=model,
                device=device,
                classes=classes,
                is_test=False
            )

            epoch_status_message += f"| Pre@1: {val_info['precision_at_1']:.6f} " \
                                    f"| R-Pre: {val_info['r_precision']:.6f} " \
                                    f"| MAP@R: {val_info['mean_average_precision_at_r']:.6f} " \
                                    f"| Sil: {val_info['silhouette']:.3f} " \
                                    f"| CH: {val_info['ch']:.0f} "
        else:
            raise Exception("Invalid task type, should either by 'classification' or 'metric'")

        epoch_time = time.time() - epoch_start_time

        print(epoch_status_message + f"| Duration: {epoch_time:.2f}s | ------------")

        # Save run info
        learn_info[epoch_idx] = {
            'train_info': train_info,
            'val_info': val_info,
            'time': epoch_time,
            'learning_rate': f'{curr_lr:g}',
        }

        # Save a checkpoint when the epoch finishes
        state = {'epoch': epoch_idx, 'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
        file_path = f'./saves/checkpoints/checkpoint_{epoch_idx}.pth'
        torch.save(state, file_path)

        if lr_lim is not None:
            if prev_val_acc is not None and prev_val_acc >= val_info['accuracy']:
                bad_val_counter += 1

                if bad_val_counter >= bad_val_lim and curr_lr > lr_lim:
                    # Step the learning rate
                    scheduler.step()
                    curr_lr = optimizer.param_groups[0]['lr']
                    bad_val_counter = 0
                    bad_val_lim = 5
            else:
                prev_val_acc = val_info['accuracy']
        else:
            scheduler.step()

    print(f'| Finished learning | Learning time: {(time.time()-learn_start_time):.2f}s')

    return model, learn_info
