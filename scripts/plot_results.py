"""Plot one axial run and export its actual metrics and history to TensorBoard."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, default=Path('artifacts/retained_models/axial_transformer'))
    parser.add_argument('--tensorboard-dir', type=Path, default=Path('artifacts/tensorboard'))
    args = parser.parse_args()
    folder = args.run_dir.resolve()
    report = json.loads((folder / 'final_test.json').read_text())
    history = json.loads((folder / 'model.history.json').read_text())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image
    from torch.utils.tensorboard import SummaryWriter

    metrics = report['metrics']
    labels = report['confusion_matrix']['truth_labels']
    columns = report['confusion_matrix']['predicted_labels']
    matrix = np.asarray(report['confusion_matrix']['counts'])
    # Include error/rejection columns only when they contain records.
    selected = [i for i, label in enumerate(columns) if label in labels or matrix[:, i].any()]
    visible = matrix[:, selected]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.imshow(visible, cmap='Blues', vmin=0, vmax=max(1, int(visible.max())))
    ax.set_xticks(range(len(selected)), [columns[i] for i in selected], rotation=60, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set(title=f"Identification: {metrics['accuracy']:.1%} ({metrics['records']} queries)",
           xlabel='Predicted identity', ylabel='True identity')
    for i, j in zip(*np.nonzero(visible)):
        ax.text(j, i, str(visible[i, j]), ha='center', va='center')
    axes[1].plot([r['epoch'] for r in history], [r['training_loss'] for r in history], color='#16a34a')
    axes[1].set(xlabel='Epoch', ylabel='Training loss', title='Axial encoder training')
    axes[1].grid(alpha=.2)
    fig.tight_layout()
    image_path = folder / 'summary.png'
    fig.savefig(image_path, dpi=180)
    fig.savefig(folder / 'summary.pdf')
    plt.close(fig)
    with SummaryWriter(str(args.tensorboard_dir / 'run'), purge_step=0) as writer:
        for row in history:
            for key in ('training_loss', 'metric_loss', 'consistency_loss'):
                if key in row:
                    writer.add_scalar('train/' + key, row[key], row['epoch'])
        writer.add_scalar('evaluation/accuracy', metrics['accuracy'], 0)
        writer.add_scalar('evaluation/records', metrics['records'], 0)
        writer.add_scalar('evaluation/failed_records', metrics['failed_records'], 0)
        writer.add_text('protocol', 'A single evaluation result, not a validation curve. Use disjoint training and target identities and independent enrollment/query sessions.', 0)
        with Image.open(image_path) as image:
            writer.add_image('evaluation/summary', np.asarray(image.convert('RGB')).copy(), 0, dataformats='HWC')
    for path in sorted((folder / 'development').glob('fold_*.json')):
        development = json.loads(path.read_text())
        with SummaryWriter(str(args.tensorboard_dir / 'development' / path.stem), purge_step=0) as writer:
            for row in development['history']:
                writer.add_scalar('train/loss', row['training_loss'], row['epoch'])
                if 'development' in row:
                    writer.add_scalar('development/accuracy', row['development']['metrics']['accuracy'], row['epoch'])
    print(image_path)


if __name__ == '__main__':
    main()
