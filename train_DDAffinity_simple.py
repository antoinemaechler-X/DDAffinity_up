import os
import sys
import shutil
import argparse
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.tensorboard
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from rde.utils.misc import BlackHole, load_config, seed_all, get_logger, get_new_dir, current_milli_time
from rde.models.protein_mpnn_network_2 import ProteinMPNN_NET
from rde.utils.skempi_mpnn_simple import SkempiDatasetManagerSimple
from rde.utils.transforms import get_transform
from rde.utils.train_mpnn import *
from rde.utils.early_stopping import EarlyStopping

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--logdir', type=str, default='./logs_skempi_simple')
    parser.add_argument('--early_stoppingdir', type=str, default='./early_stopping_simple')
    parser.add_argument('--debug', action='store_true', default=False)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--tag', type=str, default='')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--test_size', type=float, default=0.1)
    parser.add_argument('--random_state', type=int, default=42)
    parser.add_argument('--reset_cache', action='store_true', help='Reset the LMDB cache before training')
    args = parser.parse_args()

    # Load configs
    config, config_name = load_config(args.config)
    seed_all(config.train.seed)

    # Logging
    if args.debug:
        logger = get_logger('train', None)
        writer = BlackHole()
        ckpt_dir = None
    else:
        if args.resume:
            log_dir = get_new_dir(args.logdir, tag=args.tag)
        else:
            log_dir = get_new_dir(args.logdir, prefix=f'[test{args.test_size}]', tag=args.tag)
            early_stoppingdir = get_new_dir(args.early_stoppingdir, prefix=f'[test{args.test_size}]', tag=args.tag)

        ckpt_dir = os.path.join(log_dir, 'checkpoints')
        if not os.path.exists(ckpt_dir): os.makedirs(ckpt_dir)
        logger = get_logger('train', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        if not os.path.exists(os.path.join(log_dir, os.path.basename(args.config))):
            shutil.copyfile(args.config, os.path.join(log_dir, os.path.basename(args.config)))
    logger.info(args)
    logger.info(config)

    # Data
    logger.info('Loading datasets...')
    dataset_mgr = SkempiDatasetManagerSimple(
        config,
        num_workers=args.num_workers,
        logger=logger,
        test_size=args.test_size,
        random_state=args.random_state,
        reset_cache=args.reset_cache
    )

    # Model, Optimizer & Early Stopping
    logger.info('Building model...')
    model = ProteinMPNN_NET(config).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.train.lr,
        weight_decay=config.train.weight_decay
    )
    early_stopping = EarlyStopping(
        save_path=early_stoppingdir,
        patience=config.train.early_stopping.patience,
        verbose=config.train.early_stopping.verbose,
        delta=config.train.early_stopping.delta
    )
    it_first = 0

    # Resume
    if args.resume is not None:
        logger.info('Resuming from checkpoint: %s' % args.resume)
        ckpt = torch.load(args.resume, map_location=args.device)
        it_first = ckpt['iteration']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])

    def train_one_epoch(epoch):
        if early_stopping.early_stop:
            return

        time_start = current_milli_time()
        mean_loss = torch.zeros(1).to(args.device)
        model.train()
        train_loader = dataset_mgr.get_train_loader()
        train_loader = tqdm(train_loader, file=sys.stdout)

        for step, data in enumerate(train_loader):
            batch = recursive_to(data, args.device)

            # Forward pass
            loss_dict, _ = model(batch)
            loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
            time_forward_end = current_milli_time()

            # Backward
            loss.backward()

            mean_loss = (mean_loss * step + loss.detach()) / (step + 1)
            train_loader.desc = f"[epoch {epoch}] mean loss {round(mean_loss.item(), 3)}"

            orig_grad_norm = clip_grad_norm_(model.parameters(), config.train.max_grad_norm)

            optimizer.step()
            optimizer.zero_grad()

        time_backward_end = current_milli_time()
        logger.info(f'[epoch {epoch}] mean loss {mean_loss.item():.3f}')

        if epoch >= config.train.max_epochs // 2 and not early_stopping.early_stop:
            early_stopping(mean_loss.item(), model, 0)  # Using fold 0 since we're not doing cross-validation

    def test(epoch):
        scalar_accum = ScalarMetricAccumulator()
        results = []
        with torch.no_grad():
            model.eval()
            for i, batch in enumerate(tqdm(dataset_mgr.get_test_loader(), desc='Test', dynamic_ncols=True)):
                batch = recursive_to(batch, args.device)
                loss_dict, output_dict = model(batch)
                loss = sum_weighted_losses(loss_dict, config.train.loss_weights)
                scalar_accum.add(name='loss', value=loss, batchsize=batch['size'], mode='mean')

                for complex, mutstr, ddg_true, ddg_pred in zip(
                    batch["wt"]['complex'], 
                    batch["wt"]['mutstr'], 
                    output_dict['ddG_true'], 
                    output_dict['ddG_pred']
                ):
                    results.append({
                        'complex': complex,
                        'mutstr': mutstr,
                        'num_muts': len(mutstr.split(',')),
                        'ddG': ddg_true.item(),
                        'ddG_pred': ddg_pred.item()
                    })

        results = pd.DataFrame(results)
        if ckpt_dir is not None:
            results.to_csv(os.path.join(ckpt_dir, f'results_{epoch}.csv'), index=False)
        
        pearson = results[['ddG', 'ddG_pred']].corr('pearson').iloc[0, 1]
        spearman = results[['ddG', 'ddG_pred']].corr('spearman').iloc[0, 1]

        logger.info(f'[Test] Pearson {pearson:.6f} Spearman {spearman:.6f}')
        avg_loss = scalar_accum.get_average('loss')
        scalar_accum.log(epoch, 'test', logger=logger, writer=writer)
        writer.add_scalar('test/pearson', pearson, epoch)
        writer.add_scalar('test/spearman', spearman, epoch)

        return avg_loss

    try:
        for epoch in range(it_first, config.train.max_epochs + 1):
            train_one_epoch(epoch)

            if epoch % config.train.val_freq == 0:
                avg_loss = test(epoch)

        # Final evaluation and save
        logger.info('Saving final model...')
        ckpt = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'iteration': config.train.max_epochs,
            'config': config
        }
        torch.save(ckpt, os.path.join(ckpt_dir, 'DDAffinity_simple.pt'))

        # Final test evaluation
        results = []
        model.eval()
        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataset_mgr.get_test_loader(), desc='Final Test', dynamic_ncols=True)):
                batch = recursive_to(batch, args.device)
                loss_dict, output_dict = model(batch)
                for complex, mutstr, ddg_true, ddg_pred in zip(
                    batch["wt"]['complex'], 
                    batch["wt"]['mutstr'], 
                    output_dict['ddG_true'], 
                    output_dict['ddG_pred']
                ):
                    results.append({
                        'complex': complex,
                        'mutstr': mutstr,
                        'num_muts': len(mutstr.split(',')),
                        'ddG': ddg_true.item(),
                        'ddG_pred': ddg_pred.item()
                    })

        results = pd.DataFrame(results)
        results.to_csv(os.path.join(ckpt_dir, 'results_final.csv'), index=False)
        pearson = results[['ddG', 'ddG_pred']].corr('pearson').iloc[0, 1]
        spearman = results[['ddG', 'ddG_pred']].corr('spearman').iloc[0, 1]
        logger.info(f'[Final Test] Pearson {pearson:.6f} Spearman {spearman:.6f}')

    except KeyboardInterrupt:
        logger.info('Terminating...') 