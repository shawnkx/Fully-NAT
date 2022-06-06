import re
import sys
import os

checkpoint_dir = sys.argv[1]
output = sys.argv[2]
ckpt_files = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if re.match(r'checkpoint\.best*', f)]
cmd = 'python scripts/average_checkpoints.py --inputs ' + ' '.join(ckpt_files) + ' --output ' + checkpoint_dir + '/{}'.format(output)
print(cmd)
os.system(cmd)
