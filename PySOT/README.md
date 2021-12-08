# INSTRUCTIONS:

1) clone the [PySOT](https://github.com/STVIR/pysot) and create appropriate environment.
2) put file `sequences` (this is TOTB dataset) into `testing _dataset`. 
3) put `sequences.json` into `testing_dataset`
4) merge `experiments` file with `experiments` in PySOT
5) download appropriate models from [here](https://drive.google.com/drive/folders/1uOF84nH0oBV24Xki2dx6zdzf_MV4Q8QY?usp=sharing).
6) go into the file for the model you want to run experiments, for example `pysot/experiments/siammask_r50_l3` and run the following command
```bash
python -u ../../tools/test.py --snapshot model.pth --dataset sequences --config config.yaml
```
