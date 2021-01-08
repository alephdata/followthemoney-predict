NOTES:

- Some trash samples are really just trash where punctuations or tables weren't properlly OCR'd... is there a way to filter them?
    - meetagain
    - 2348927342983*2341/234

- Best setup: Train a model on public data, quantize/fine-tune with private:
    model_types_public = fasttext.load_model('./data.public/models/types/model.bin')
    model_types_public.quantize(verbose=True, input='./data.secret/model_data/types/train.txt', retrain=True, qnorm=True)
