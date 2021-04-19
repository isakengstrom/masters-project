import time
import math

import torch.optim.lr_scheduler


from train import train
from evaluate import evaluate


def learn(train_loader, val_loader, model, optimizer, loss_function, num_epochs, device, network_type="triplet"):

    learn_start_time = time.time()

    total_accuracy = None
    accuracy_log = []
    loss_log = []

    num_epochs_digits = int(math.log10(num_epochs)) + 1

    step_size = 1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size, gamma=.1)
    curr_lr = optimizer.param_groups[0]['lr']
    prev_lr = curr_lr

    for epoch_idx in range(1, num_epochs + 1):
        epoch_start_time = time.time()

        print(f"| Epoch {epoch_idx:{num_epochs_digits}.0f}/{num_epochs} "
              f"| {'Updated l' if curr_lr != prev_lr else 'L'}earning rate: {curr_lr}")

        model = train(
            data_loader=train_loader,
            model=model,
            optimizer=optimizer,
            loss_function=loss_function,
            device=device,
            network_type=network_type,
            epoch_idx=epoch_idx,
            num_epochs=num_epochs
        )

        val_accuracy = evaluate(
            data_loader=val_loader,
            model=model,
            device=device
        )

        if total_accuracy is not None and total_accuracy > val_accuracy:
            if curr_lr > 5e-12:
                prev_lr = curr_lr

                # Step the learning rate
                scheduler.step()
                curr_lr = optimizer.param_groups[0]['lr']
        else:
            total_accuracy = val_accuracy

        # Print info from the finished epoch
        #print('-' * 72)
        print(f'| Epoch {epoch_idx:{num_epochs_digits}.0f}/{num_epochs} '
              f'| Duration: {time.time()-epoch_start_time:.2f}s '
              f'| Val accuracy: {val_accuracy:.3f} |')
        print('-' * 72)

        # Save a checkpoint when the epoch finishes
        state = {'epoch': epoch_idx, 'net': model.state_dict(), 'optimizer': optimizer.state_dict()}
        file_path = f'./checkpoints/checkpoint_{epoch_idx}.pth'
        torch.save(state, file_path)

    print(f'| Finished learning | Learning time: {(time.time()-learn_start_time):2f}s')

    return model #, loss_log, acc_log