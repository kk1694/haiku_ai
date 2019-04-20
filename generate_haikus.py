
from fastai.text import *
import argparse



parser = argparse.ArgumentParser(description='Get some AI written haikus')
parser.add_argument("n", help="number of haikus to generate", type=int)
parser.add_argument("--model_export", help="file path for the exported learner", type=str)
parser.add_argument("--temp", help="temperature", type=float)
parser.add_argument('--save', help='If given, saves the output at this path. Otherwise prints', type=str)


args = parser.parse_args()
fn = Path('data/final_export' if not args.model_export else args.model_export)

learn = load_learner(fn.parent, fn.stem)

def create_haiku(n=35, temp=0.5):
    txt = learn.predict('', n, temperature=temp).replace('\n ', '\n')
    return txt.split('xxeos')[0].replace('xxbos ', '')

def create_haikus(n, temp):
    return '\n\n'.join([create_haiku(temp=temp) for _ in range(n)])

if args.save:
    txt = create_haikus(args.n, temp = args.temp if args.temp else 0.5)
    file = open(args.save, 'a')
    file.write(txt)
    file.close()    
else:
    print(create_haikus(args.n, temp = args.temp if args.temp else 0.5))