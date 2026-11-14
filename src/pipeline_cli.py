"""CLI implementations shared by train.py, detect.py and evaluate.py."""
import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from tracking import DetectionTracker
from dataset_utils import PROJECT, NAMES, resolve, file_hash, write_json, dataset_config, validate_root, absolute_config

DEFAULT_BEST = 'runs/detect/runs/birdsai_yolov8n-2/weights/best.pt'


def runtime(device):
    # Disable optional Ultralytics connections for local reproducible work.
    os.environ['YOLO_OFFLINE']='true'
    os.environ['YOLO_AUTOINSTALL']='false'
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.events import events
    events.enabled = False
    if device.startswith('xpu') and not torch.xpu.is_available():
        raise RuntimeError('Intel XPU unavailable. Use the tested XPU environment or --device cpu.')
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable. Intel Arc uses --device xpu:0, not CUDA.')
    return torch, YOLO


def require_model(path):
    path=resolve(path)
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f'Model checkpoint missing or empty: {path}. Supply --model explicitly.')
    return path


def name_now(prefix):
    return prefix+'_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')


def predict_main(default_source=None):
    p=argparse.ArgumentParser(description='Predict with the fine-tuned animal/human model; CPU is the default.')
    p.add_argument('--source', default=default_source, required=default_source is None, help='Local image, video or image-sequence directory')
    p.add_argument('--model',default=DEFAULT_BEST)
    p.add_argument('--device',default='cpu')
    p.add_argument('--conf',type=float,default=.25)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--track', action='store_true', help='Assign IDs within one ordered camera sequence or video')
    args=p.parse_args()
    tracker = DetectionTracker() if args.track else None
    if not 0 < args.conf < 1:
        p.error('--conf must be between zero and one')
    source=resolve(args.source)
    if not source.exists():
        raise FileNotFoundError(f'Prediction source missing: {source}')
    checkpoint=require_model(args.model)
    torch,YOLO=runtime(args.device)
    model=YOLO(str(checkpoint))
    # stream=True prevents retaining every frame/result of a long sequence.
    output=PROJECT/'outputs/predictions'/name_now('prediction')
    count=0
    output.mkdir(parents=True,exist_ok=False)
    with_stream=model.predict(source=str(source),device=args.device,conf=args.conf,imgsz=args.imgsz,
                              save=True,stream=True,project=str(output.parent),name=output.name,exist_ok=True)
    with (output/'detections.jsonl').open('w',encoding='utf-8') as f:
        for result in with_stream:
            boxes=[{'class':result.names[int(b.cls.item())],'confidence':float(b.conf.item()),
                    'xyxy':b.xyxy.cpu().tolist()[0]} for b in result.boxes]
            if tracker is not None:
                boxes = tracker.update(boxes)
            f.write(json.dumps({'source':result.path,'detections':boxes})+'\n')
            count+=1
    write_json(output/'run.json',{'checkpoint':str(checkpoint),'checkpoint_sha256':file_hash(checkpoint),
               'source':str(source),'device':args.device,'torch':torch.__version__,
               'confidence':args.conf,'imgsz':args.imgsz,'frames':count,'tracking':args.track})
    print(f'Prediction complete: {output}')


def operating_matrix(names,confidence=.25,match_iou=.5):
    from ultralytics.utils.metrics import ConfusionMatrix
    class OperatingMatrix(ConfusionMatrix):
        def process_batch(self,detections,batch,conf=.25,iou_thres=.45):
            return super().process_batch(detections,batch,conf=confidence,iou_thres=match_iou)
    return OperatingMatrix(names=names,save_matches=False)


def operating_validator(confidence=.25, match_iou=.5):
    """Keep AP at 0.001 while plotting a matrix at a documented operating threshold."""
    from ultralytics.models.yolo.detect.val import DetectionValidator

    class OperatingValidator(DetectionValidator):
        def init_metrics(self,model):
            super().init_metrics(model)
            self.confusion_matrix=operating_matrix(model.names,confidence,match_iou)
    return OperatingValidator


def evaluate_main():
    p=argparse.ArgumentParser(description='Evaluate a fixed checkpoint on annotated val/test frames.')
    p.add_argument('--data',default='configs/yolo_dataset.yaml')
    p.add_argument('--model',default=DEFAULT_BEST)
    p.add_argument('--split',choices=('val','test'),default='test')
    p.add_argument('--device',default='cpu')
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--imgsz',type=int,default=640)
    p.add_argument('--cm-conf',type=float,default=.25)
    p.add_argument('--match-iou',type=float,default=.5)
    args=p.parse_args()
    if args.batch < 1 or not 0 < args.cm_conf < 1 or not 0 < args.match_iou <= 1:
        p.error('Invalid batch, confidence or matching IoU')
    root,cfg=dataset_config(args.data)
    if args.split not in cfg:
        raise ValueError(f'No {args.split} split in {args.data}; prepare it first')
    inventory=validate_root(root,tuple(s for s in ('train','val','test') if s in cfg))
    checkpoint=require_model(args.model)
    checkpoint_sha=file_hash(checkpoint)
    torch,YOLO=runtime(args.device)
    output=PROJECT/'outputs/evaluation'/name_now(args.split)
    model=YOLO(str(checkpoint))
    if model.names != NAMES:
        raise ValueError(f'Evaluation requires the two-class fine-tuned checkpoint, got {model.names}. A COCO baseline needs explicit class remapping.')
    metrics=model.val(data=str(absolute_config(args.data)),split=args.split,device=args.device,
                      batch=args.batch,imgsz=args.imgsz,workers=0,plots=True,conf=.001,iou=.7,
                      validator=operating_validator(args.cm_conf,args.match_iou),
                      project=str(output.parent),name=output.name,exist_ok=False)
    if file_hash(checkpoint) != checkpoint_sha:
        raise RuntimeError('Checkpoint changed during evaluation; discard this run and evaluate a frozen copy')
    per_class=[]
    for i,cls in enumerate(metrics.box.ap_class_index):
        precision,recall,ap50,ap95=metrics.box.class_result(i)
        per_class.append({'class':model.names[int(cls)],'precision_at_AP_summary_point':float(precision),
                          'recall_at_AP_summary_point':float(recall),'AP50':float(ap50),'AP50_95':float(ap95)})
    matrix=metrics.confusion_matrix.matrix
    operating=[]
    for cls,label in model.names.items():
        tp=float(matrix[cls,cls]); fp=float(matrix[cls,:].sum()-tp); fn=float(matrix[:,cls].sum()-tp)
        operating.append({'class':label,'TP':tp,'FP':fp,'FN':fn,
                          'precision':tp/(tp+fp) if tp+fp else 0.,'recall':tp/(tp+fn) if tp+fn else 0.})
    result={'checkpoint':str(checkpoint),'checkpoint_sha256':checkpoint_sha,'dataset_root':str(root),
            'dataset_manifest_sha256':file_hash(root/'manifest.json') if (root/'manifest.json').exists() else None,
            'inventory':inventory,'split':args.split,'torch':torch.__version__,'device':args.device,
            'imgsz':args.imgsz,'AP_confidence_floor':.001,'NMS_IoU':.7,
            'confusion_matrix_confidence':args.cm_conf,'confusion_matrix_matching_IoU':args.match_iou,
            'metrics':{k:float(v) for k,v in metrics.results_dict.items()},'per_class_AP':per_class,
            'confusion_matrix':matrix.tolist(),'per_class_operating_point':operating,'speed_ms':metrics.speed,
            'scope':'Annotated frames only. AP summary P/R and fixed-threshold matrix P/R use different operating points.'}
    write_json(output/'test_results.json',result)
    for suffix in ('','_normalized'):
        path=output/f'confusion_matrix{suffix}.png'
        if path.exists():
            path.rename(output/f'confusion_matrix{suffix}_conf{args.cm_conf:.2f}_iou{args.match_iou:.2f}.png')
    print(json.dumps(result['metrics'],indent=2))
    print(f'Evaluation complete: {output}')


def train_main():
    p=argparse.ArgumentParser(description='Train on a verified sequence-disjoint dataset; never resume silently.')
    p.add_argument('--data',default='configs/yolo_dataset.yaml')
    p.add_argument('--model',default='yolov8n.pt',help='Initial pretrained checkpoint for a fresh run')
    p.add_argument('--resume',help='Explicit last.pt from a previous corrected run')
    p.add_argument('--device',default='cpu')
    p.add_argument('--batch',type=int,default=2)
    p.add_argument('--epochs',type=int,default=50)
    p.add_argument('--patience', type=int, default=100, help='Epochs without validation improvement before early stopping')
    p.add_argument('--workers',type=int,default=0)
    p.add_argument('--imgsz',type=int,default=640)
    args=p.parse_args()
    if min(args.batch,args.epochs,args.imgsz) < 1 or min(args.workers, args.patience) < 0:
        p.error('Invalid training settings')
    root,cfg=dataset_config(args.data)
    if not {'train','val'} <= cfg.keys():
        raise ValueError('Training YAML must define both train and val splits')
    inventory=validate_root(root,tuple(s for s in ('train','val','test') if s in cfg))
    manifest=root/'manifest.json'
    if not manifest.is_file():
        raise ValueError('Training requires manifest.json from the corrected preparation pipeline')
    dataset_sha=file_hash(manifest)
    checkpoint=require_model(args.resume or args.model)
    if args.resume:
        previous=checkpoint.parent.parent/'run_manifest.json'
        if previous.exists():
            if json.loads(previous.read_text())['dataset_manifest_sha256'] != dataset_sha:
                raise ValueError('Resume dataset differs from original run; start a fresh run')
        else:
            raise ValueError('Legacy checkpoint has no dataset provenance. Do not resume it on a rebuilt split; start fresh from COCO weights.')
    torch,YOLO=runtime(args.device)
    model=YOLO(str(checkpoint))
    provenance={'dataset_manifest_sha256':dataset_sha,'dataset_root':str(root),'inventory':inventory,
                'initial_checkpoint':str(checkpoint),'initial_checkpoint_sha256':file_hash(checkpoint),
                'torch':torch.__version__,'device':args.device}
    def record(trainer):
        write_json(Path(trainer.save_dir)/'run_manifest.json',provenance)
    model.add_callback('on_pretrain_routine_end',record)
    options={'data':str(absolute_config(args.data)),'device':args.device,'batch':args.batch,
             'workers':args.workers,'imgsz':args.imgsz,'plots':True}
    if args.resume:
        options['resume']=str(checkpoint)
    else:
        options.update(epochs=args.epochs,patience=args.patience,project=str(PROJECT/'runs'),name=name_now('birdsai_corrected'),amp=False)
    model.train(**options)
