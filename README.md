# InkCloser

**InkCloser: Line-Art Completion for Paint-Bucket Colorization**

Paint-bucket (flood-fill) coloring of line art only works when the drawing forms closed regions; a single small gap in a stroke lets the fill leak into neighboring areas. InkCloser repairs line art automatically by restoring the missing ink, so a standard paint-bucket tool can be used without manual retouching.

Unlike conventional inpainting, the location of the gaps is not known in advance: the model has to find and close them on its own while leaving the rest of the drawing, including intentional open stroke endings, unchanged. InkCloser is a U-Net with gated convolutions, a dilated context stack and transformer self-attention, trained on pairs built by synthetically removing ink segments from clean, closed line art (coloring pages).

## Dataset

The InkCloser Dataset (55,000 pairs of corrupted and clean line art) is available [here](https://drive.google.com/file/d/1NO8i_J5mSlh7YYuwh2ZyFqkIW7ISrGuk/view?usp=drive_link).

## Code

- [`inkrepair_datasetgenerator_v2.ipynb`](inkrepair_datasetgenerator_v2.ipynb): builds the training dataset by removing line segments from clean line art and creating the train/test split.
- [`lineartanimdata3_closer_v12.ipynb`](lineartanimdata3_closer_v12.ipynb): defines, trains and evaluates the InkCloser model.

## Paper

If you use this code, please cite:

```bibtex
@article{TODO,
  title   = {InkCloser: Line-Art Completion for Paint-Bucket Colorization},
  author  = {TODO},
  journal = {TODO},
  year    = {TODO},
  volume  = {TODO},
  pages   = {TODO},
  doi     = {TODO}
}
```
