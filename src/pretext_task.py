import argparse, tqdm, torch, os, logging

from dataset import get_dataset
from simclr import ContrastiveLoss
from model import PreModel
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from utils import save_config_file, save_checkpoint, accuracy

parser = argparse.ArgumentParser(description='PyTorch SimCLR')
parser.add_argument('--device', type=str, default='cuda:1')

parser.add_argument('--batch_size', type=int, default=1024)
parser.add_argument('--num_epochs', type=int, default=200)
parser.add_argument('--lr', type=int, default=0.0003)
parser.add_argument('--weight_decay', type=float, default=1e-4)

parser.add_argument('--temperature', type=float, default=0.07)
parser.add_argument('--n_views', type=int, default=2)

parser.add_argument('--log_every_n_steps', type=int, default=100)


def main():
    args = parser.parse_args()

    train_dataset = get_dataset(root_folder='./data', name='cifar10', n_views=args.n_views)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2,
        pin_memory=True,
    )

    model = PreModel()
    model.train()
    model.to(args.device)

    optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
    criterion = torch.nn.CrossEntropyLoss()
    contrastive_loss = ContrastiveLoss(batch_size=args.batch_size, n_views=args.n_views, temperature=args.temperature, logits=True)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(train_loader), eta_min=0, last_epoch=-1)
 
    writer = SummaryWriter()
    logging.basicConfig(filename=os.path.join(writer.log_dir, 'training.log'), level=logging.DEBUG)
    save_config_file(writer.log_dir, args)

    scaler = GradScaler()

    n_iter = 0
    for epoch_counter in range(args.num_epochs):
        for images, _ in tqdm.tqdm(train_loader): ## images (n_views x batch_size x channels x width x height)
            images = torch.cat(images, dim=0) ## images (2 x batch_size x channels x width x height)
            images = images.to(args.device)

            with autocast():
                features = model(images)
                logits, labels = contrastive_loss(features)
                loss = criterion(logits, labels)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if n_iter % args.log_every_n_steps == 0:
                top1, top5 = accuracy(logits, labels, topk=(1, 5))
                writer.add_scalar('loss', loss, global_step=n_iter)
                writer.add_scalar('acc/top1', top1[0], global_step=n_iter)
                writer.add_scalar('acc/top5', top5[0], global_step=n_iter)
                writer.add_scalar('learning_rate', scheduler.get_last_lr()[0], global_step=n_iter)

            n_iter += 1

        if epoch_counter >= 10:
            scheduler.step()

        logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}\tTop1 accuracy: {top1[0]}")
        logging.info("Training has finished.")

        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(epoch_counter)
        save_checkpoint({
            'epoch': epoch_counter,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {writer.log_dir}.")


if __name__ == '__main__':
    main()