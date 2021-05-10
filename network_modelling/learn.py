import time
import math

import torch.optim.lr_scheduler

from train import train
from evaluate import evaluate


def learn(train_loader, val_loader, model, optimizer, loss_function, num_epochs, device, classes, tb_writer,
          loss_type="triplet"):

    learn_start_time = time.time()

    prev_val_acc = None

    epoch_formatter = int(math.log10(num_epochs)) + 1

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=.1)
    curr_lr = optimizer.param_groups[0]['lr']
    prev_lr = curr_lr
    lim_lr = 5.1e-10

    bad_val_counter = 0
    bad_val_lim = 5

    learn_info = {'epochs': dict(), 'learning_rate_lim': lim_lr}

    for epoch_idx in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        epoch_status_message = f"| Epoch {epoch_idx:{epoch_formatter}.0f}/{num_epochs} "

        if curr_lr <= lim_lr:
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
            epoch_idx=epoch_idx,
            num_epochs=num_epochs,
            classes=classes,
            tb_writer=tb_writer
        )

        # Validate the epoch
        val_info = evaluate(
            data_loader=val_loader,
            model=model,
            device=device,
            is_test=False
        )

        epoch_time = time.time() - epoch_start_time

        print(epoch_status_message +
              f"| Val accuracy: {val_info['accuracy']:.6f} "
              f"| Duration: {epoch_time:.2f}s |")
        print('-' * 72)

        # Save run info
        learn_info['epochs'][epoch_idx] = {
            'train_info': train_info,
            'val_info': val_info,
            'time': epoch_time,
            'learning_rate': f'{curr_lr:g}',
        }

        # Save a checkpoint when the epoch finishes
        state = {'epoch': epoch_idx, 'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
        file_path = f'./checkpoints/checkpoint_{epoch_idx}.pth'
        torch.save(state, file_path)

        if prev_val_acc is not None and prev_val_acc >= val_info['accuracy']:
            bad_val_counter += 1

            if bad_val_counter >= bad_val_lim and curr_lr > lim_lr:
                # Step the learning rate
                scheduler.step()
                curr_lr = optimizer.param_groups[0]['lr']
                bad_val_counter = 0
        else:
            prev_val_acc = val_info['accuracy']

    print(f'| Finished learning | Learning time: {(time.time()-learn_start_time):.2f}s')

    return model, learn_info
