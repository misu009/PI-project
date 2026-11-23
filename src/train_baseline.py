import argparse
import os
import time

from datetime import datetime
from timeit import default_timer

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn import metrics

import seed
from sms import send
import config

from transformations.mixup import mixup_volume


# ==========================================================
# ARGUMENTE
# ==========================================================

def parse_arguments():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-t', '--task',
        type=str,
        required=True,
        choices=['abnormal', 'acl', 'meniscus']
    )

    parser.add_argument(
        '-p', '--plane',
        type=str,
        required=True,
        choices=['sagittal', 'coronal', 'axial']
    )

    parser.add_argument(
        '--prefix_name',
        type=str,
        required=True
    )

    parser.add_argument(
        '--model',
        type=str,
        default='seresnet50'
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=100
    )

    parser.add_argument(
        '--cut',
        type=str,
        default='vertical'
    )

    parser.add_argument(
        '--augment',
        type=str,
        default='albumentations-group'
    )

    parser.add_argument(
        '--augment_prob',
        type=float,
        default=0.5
    )

    # MIXUP
    parser.add_argument(
        '--mixup_alpha',
        type=float,
        default=0.0,
        help='0 dezactiveaza Mixup; 0.4 il activeaza'
    )

    parser.add_argument(
        '--mixup_prob',
        type=float,
        default=1.0,
        help='Probabilitatea aplicarii Mixup'
    )

    parser.add_argument(
        '--lr_scheduler',
        type=str,
        choices=['plateau', 'step'],
        default='plateau'
    )

    parser.add_argument(
        '--save_model',
        type=int,
        choices=[0, 1],
        default=1
    )

    parser.add_argument(
        '--log_every',
        type=int,
        default=100
    )

    return parser.parse_args()


# ==========================================================
# DATALOADER
# ==========================================================

def collate_fn(batch):
    """
    Pastreaza volumele cu numar variabil de felii
    separat, fara sa le combine intr-un singur tensor.
    """
    return tuple(zip(*batch))


# ==========================================================
# ANTRENARE PRINCIPALA
# ==========================================================

def run(args):

    if args.mixup_alpha < 0:
        raise ValueError(
            '--mixup_alpha trebuie sa fie >= 0'
        )

    if not 0 <= args.mixup_prob <= 1:
        raise ValueError(
            '--mixup_prob trebuie sa fie intre 0 si 1'
        )

    print(
        f'[MIXUP] alpha={args.mixup_alpha}, '
        f'prob={args.mixup_prob}',
        flush=True
    )

    start = default_timer()

    # GPU / CPU
    device = torch.device(
        'cuda' if torch.cuda.is_available() else 'cpu'
    )

    use_amp = device.type == 'cuda'

    scaler = torch.amp.GradScaler(
        'cuda',
        enabled=use_amp
    )

    # LOGGING
    log_root_folder = './logs/'

    now = datetime.now()

    logdir = os.path.join(
        log_root_folder,
        '{}_{}_{}'.format(
            now.strftime('%Y%m%d-%H%M%S'),
            args.task,
            args.plane
        )
    )

    os.makedirs(logdir, exist_ok=True)

    MAIN_LOG = os.path.join(
        logdir,
        'mainlog.txt'
    )

    with open(MAIN_LOG, 'w') as f:
        print(
            f"""MRNET
Experiment: {now.strftime('%d%m%Y-%H%M%S')}
Task: {args.task}
Plane: {args.plane}
Prefix: {args.prefix_name}
Model: {args.model}
Device: {device}
Epochs: {args.epochs}
Augmentation: {args.augment}
Mixup Alpha: {args.mixup_alpha}
Mixup Probability: {args.mixup_prob}
Learning Rate: 1e-5
""",
            file=f
        )

    from torch.utils.tensorboard import SummaryWriter

    writer = SummaryWriter(logdir)

    # DATASET
    from loaders.loader_baseline import MRDataset

    train_dataset = MRDataset(
        args.task,
        args.plane,
        train_val_test='train',
        cut=args.cut,
        augment=args.augment,
        augment_prob=args.augment_prob
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=use_amp,
        persistent_workers=True
    )

    validation_dataset = MRDataset(
        args.task,
        args.plane,
        train_val_test='valid',
        cut=args.cut,
        augment=args.augment,
        augment_prob=args.augment_prob
    )

    validation_loader = torch.utils.data.DataLoader(
        validation_dataset,
        batch_size=8,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        num_workers=2,
        pin_memory=use_amp,
        persistent_workers=True
    )

    # MODEL
    from models import choose

    model = choose.get_model(args.model)

    # Aceleasi greutati initiale ca la baseline
    initial_weights = os.environ.get(
        'RMN_INIT_WEIGHTS'
    )

    if initial_weights:
        print(
            '[TRAIN] Incarcam greutatile initiale:',
            initial_weights,
            flush=True
        )

        state_dict = torch.load(
            initial_weights,
            map_location='cpu',
            weights_only=True
        )

        model.load_state_dict(
            state_dict,
            strict=True
        )

    model.to(device)

    # OPTIMIZER
    optimizer = optim.Adam(
        model.parameters(),
        lr=1e-5,
        weight_decay=0.1
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        patience=3,
        factor=0.3,
        threshold=1e-4
    )

    best_val_auc = 0.0

    # Folder pentru modele
    os.makedirs(
        config.TRAIN_MODELS_PATH_PRETRAINED,
        exist_ok=True
    )

    print(
        '[TRAIN] Starting Training!',
        flush=True
    )

    # ======================================================
    # EPOCI
    # ======================================================

    for epoch in range(args.epochs):

        print(
            f'\n[TRAIN] EPOCH {epoch + 1}/{args.epochs}',
            flush=True
        )

        current_lr = get_lr(optimizer)

        t_start = time.time()

        train_loss, train_auc = train_model(
            model,
            train_loader,
            device,
            epoch,
            args.epochs,
            optimizer,
            writer,
            MAIN_LOG,
            current_lr,
            args.log_every,
            scaler,
            use_amp,
            mixup_alpha=args.mixup_alpha,
            mixup_prob=args.mixup_prob
        )

        val_loss, val_auc = evaluate_model(
            model,
            validation_loader,
            device,
            epoch,
            args.epochs,
            writer,
            MAIN_LOG,
            current_lr,
            use_amp
        )

        if args.lr_scheduler == 'plateau':
            scheduler.step(val_loss)

        delta = (
            time.time() - t_start
        ) / 60.0

        train_auc_text = (
            'N/A (Mixup)'
            if train_auc is None
            else f'{train_auc:.4f}'
        )

        statement = (
            f'Epoch {epoch + 1}:\n'
            f'Train Loss: {train_loss:.4f} | '
            f'Train AUC: {train_auc_text}\n'
            f'Val Loss: {val_loss:.4f} | '
            f'Val AUC: {val_auc:.4f}\n'
            f'Elapsed time: {delta:.2f} min'
        )

        print(statement, flush=True)

        with open(MAIN_LOG, 'a') as f:
            print(statement, file=f)

        writer.add_scalar(
            'Epoch/Train_Loss',
            train_loss,
            epoch
        )

        writer.add_scalar(
            'Epoch/Validation_Loss',
            val_loss,
            epoch
        )

        writer.add_scalar(
            'Epoch/Validation_AUC',
            val_auc,
            epoch
        )

        # ==================================================
        # SALVAREA CELUI MAI BUN MODEL
        # ==================================================

        if val_auc > best_val_auc:

            best_val_auc = val_auc

            if bool(args.save_model):

                train_auc_filename = (
                    'mixup-na'
                    if train_auc is None
                    else f'{train_auc:.4f}'
                )

                filename = (
                    f'model_{args.prefix_name}_'
                    f'{args.task}_{args.plane}_'
                    f'train_auc_{train_auc_filename}_'
                    f'val_auc_{val_auc:.4f}_'
                    f'train_loss_{train_loss:.4f}_'
                    f'val_loss_{val_loss:.4f}_'
                    f'epoch_{epoch + 1}_'
                    f'arch_{args.model}_'
                    f'mixup_alpha_{args.mixup_alpha}_'
                    f'mixup_prob_{args.mixup_prob}.pth'
                )

                model_path = os.path.join(
                    config.TRAIN_MODELS_PATH_PRETRAINED,
                    filename
                )

                torch.save(
                    model.state_dict(),
                    model_path
                )

                print(
                    '[TRAIN] Model saved:',
                    model_path,
                    flush=True
                )

        print('-' * 80, flush=True)

    # ======================================================
    # FINALIZARE
    # ======================================================

    elapsed = (
        default_timer() - start
    ) / 60.0

    statement = (
        f'[TRAIN] Finished after {elapsed:.2f} minutes'
    )

    print(statement, flush=True)

    with open(MAIN_LOG, 'a') as f:
        print(statement, file=f)

    writer.flush()
    writer.close()


# ==========================================================
# ANTRENARE CU MIXUP OPTIONAL
# ==========================================================

def train_model(
    model,
    train_loader,
    device,
    epoch,
    num_epochs,
    optimizer,
    writer,
    log_file,
    current_lr,
    log_every,
    scaler,
    use_amp,
    mixup_alpha=0.0,
    mixup_prob=1.0
):

    model.train()

    mixup_activ = (
        mixup_alpha > 0
        and mixup_prob > 0
    )

    y_preds = []
    y_trues = []
    losses = []

    mixed_count = 0
    total_count = 0

    total_step = len(train_loader)

    for i, (
        indices,
        images,
        labels,
        weights
    ) in enumerate(train_loader):

        optimizer.zero_grad(
            set_to_none=True
        )

        batch_loss_val = 0.0

        # Procesam volumele succesiv pentru a
        # pastra mecanismul Gradient Accumulation.

        for j in range(len(images)):

            lam = 1.0
            partner = j

            should_mix = (
                mixup_activ
                and len(images) >= 2
                and np.random.random() < mixup_prob
            )

            # ==========================================
            # MIXUP
            # ==========================================

            if should_mix:

                # Alegem un alt examen din batch
                partner = np.random.randint(
                    len(images) - 1
                )

                if partner >= j:
                    partner += 1

                # Coeficient aleator din distributia Beta
                lam = float(
                    np.random.beta(
                        mixup_alpha,
                        mixup_alpha
                    )
                )

                # Mixup pe CPU, inainte de GPU
                image_cpu = mixup_volume(
                    images[j],
                    images[partner],
                    lam
                )

                mixed_count += 1

            else:
                image_cpu = images[j]

            image = image_cpu.to(
                device,
                non_blocking=True
            )

            label_a = labels[j].to(device)
            weight_a = weights[j].to(device)

            # ==========================================
            # FORWARD + LOSS
            # ==========================================

            with torch.amp.autocast(
                device.type,
                enabled=use_amp
            ):

                prediction = model(
                    image.float()
                )[0]

                loss_a = F.binary_cross_entropy_with_logits(
                    prediction,
                    label_a,
                    weight=weight_a
                )

                if should_mix:

                    label_b = labels[partner].to(device)
                    weight_b = weights[partner].to(device)

                    loss_b = F.binary_cross_entropy_with_logits(
                        prediction,
                        label_b,
                        weight=weight_b
                    )

                    # Combinam erorile celor doua
                    # etichete folosind acelasi lambda.

                    loss = (
                        lam * loss_a
                        + (1.0 - lam) * loss_b
                    )

                else:
                    loss = loss_a

            # ==========================================
            # VERIFICARE NUMERICA
            # ==========================================

            if not torch.isfinite(loss).all():
                raise RuntimeError(
                    f'Loss invalid! '
                    f'Epoca {epoch + 1}, '
                    f'batch {i}, volum {j}'
                )

            # ==========================================
            # GRADIENT ACCUMULATION
            # ==========================================

            loss = loss / len(images)

            scaler.scale(loss).backward()

            batch_loss_val += loss.detach().item()

            total_count += 1

            # Train AUC este calculat doar cand
            # nu folosim Mixup.
            if not mixup_activ:

                probability = torch.sigmoid(
                    prediction.detach().float()
                ).item()

                y_preds.append(probability)
                y_trues.append(int(label_a.item()))

        # Actualizam greutatile dupa batch
        scaler.step(optimizer)
        scaler.update()

        losses.append(batch_loss_val)

        writer.add_scalar(
            'Train/Loss',
            batch_loss_val,
            epoch * len(train_loader) + i
        )

        # ==============================================
        # AFISARE PROGRES
        # ==============================================

        if i > 0 and i % log_every == 0:

            auc_now = None

            if (
                not mixup_activ
                and len(set(y_trues)) == 2
            ):
                auc_now = metrics.roc_auc_score(
                    y_trues,
                    y_preds
                )

            if mixup_activ:
                auc_text = 'N/A (Mixup)'
            elif auc_now is None:
                auc_text = 'N/A'
            else:
                auc_text = f'{auc_now:.4f}'

            print(
                f'Epoch [{epoch + 1}/{num_epochs}] '
                f'Step [{i}/{total_step}] '
                f'avg_train_loss: {np.mean(losses):.4f} '
                f'| train_auc: {auc_text} '
                f'| lr: {current_lr}',
                flush=True
            )

    # ======================================================
    # METRICI FINALE TRAIN
    # ======================================================

    auc = None

    if (
        not mixup_activ
        and len(set(y_trues)) == 2
    ):

        auc = float(
            metrics.roc_auc_score(
                y_trues,
                y_preds
            )
        )

        writer.add_scalar(
            'Train/AUC_epoch',
            auc,
            epoch
        )

    if mixup_activ:

        mixup_fraction = (
            mixed_count / max(total_count, 1)
        )

        writer.add_scalar(
            'Train/Mixup_fraction_epoch',
            mixup_fraction,
            epoch
        )

        print(
            f'[MIXUP] {mixed_count} din '
            f'{total_count} volume mixate',
            flush=True
        )

    train_loss_epoch = float(
        np.mean(losses)
    )

    return train_loss_epoch, auc


# ==========================================================
# EVALUARE FARA MIXUP
# ==========================================================

def evaluate_model(
    model,
    val_loader,
    device,
    epoch,
    num_epochs,
    writer,
    log_file,
    current_lr,
    use_amp,
    log_every=20
):

    model.eval()

    y_trues = []
    y_preds = []
    losses = []

    with torch.no_grad():

        for i, (
            indices,
            images,
            labels,
            weights
        ) in enumerate(val_loader):

            batch_loss = 0.0

            for j in range(len(images)):

                image = images[j].to(device)
                label = labels[j].to(device)
                weight = weights[j].to(device)

                with torch.amp.autocast(
                    device.type,
                    enabled=use_amp
                ):

                    prediction = model(
                        image.float()
                    )[0]

                    loss = F.binary_cross_entropy_with_logits(
                        prediction,
                        label,
                        weight=weight
                    )

                if not torch.isfinite(loss).all():
                    raise RuntimeError(
                        f'Validation Loss invalid! '
                        f'Epoca {epoch + 1}, batch {i}'
                    )

                batch_loss += loss.item()

                probability = torch.sigmoid(
                    prediction.float()
                ).item()

                y_preds.append(probability)
                y_trues.append(int(label.item()))

            loss_value = (
                batch_loss / len(images)
            )

            losses.append(loss_value)

            writer.add_scalar(
                'Val/Loss',
                loss_value,
                epoch * len(val_loader) + i
            )

    if len(set(y_trues)) != 2:
        raise RuntimeError(
            'Validation AUC necesita ambele clase.'
        )

    auc = metrics.roc_auc_score(
        y_trues,
        y_preds
    )

    writer.add_scalar(
        'Val/AUC_epoch',
        auc,
        epoch
    )

    val_loss_epoch = float(
        np.mean(losses)
    )

    return val_loss_epoch, float(auc)


# ==========================================================
# LEARNING RATE
# ==========================================================

def get_lr(optimizer):

    for param_group in optimizer.param_groups:
        return param_group['lr']


if __name__ == '__main__':

    args = parse_arguments()

    run(args)