import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','4')
import argparse,json

def main():
 p=argparse.ArgumentParser(description='SaudaSach research pipeline. No app or external detector API.')
 s=p.add_subparsers(dest='command',required=True)
 for name in ['audit','features','train','export','benchmark']:s.add_parser(name)
 e=s.add_parser('evaluate');e.add_argument('--final-test',action='store_true')
 i=s.add_parser('predict');i.add_argument('audio');i.add_argument('--bundle')
 a=p.parse_args()
 if a.command=='audit':
  from .data import audit;audit()
 elif a.command=='features':
  from .data import cache_features;cache_features()
 elif a.command=='train':
  from .train import train_all;train_all()
 elif a.command=='evaluate':
  from .evaluate import evaluate_all;evaluate_all(a.final_test)
 elif a.command=='export':
  from .bundle import export_bundle;export_bundle()
 elif a.command=='benchmark':
  from .bundle import deployment_benchmark;deployment_benchmark()
 elif a.command=='predict':
  from .bundle import predict
  print(json.dumps(predict(a.audio,**({'bundle':a.bundle} if a.bundle else {})),indent=2))
if __name__=='__main__':main()
