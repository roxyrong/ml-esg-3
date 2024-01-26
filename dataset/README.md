Original Files:
- ML-ESG-3_Trainset_Chinese.json
- ML-ESG-3_Trainset_English.json
- ML-ESG-3_Trainset_French.json
- ML-ESG-3_Trainset_Japanese.json
- ML-ESG-3_Trainset_Korean.json

1. aggregate all JSON files[Notebook - ml-esg-3-EDA]
- train_df.parquet

2. use DeepL translate non-English news to English and aggregate [Notebook - DeepL Translation]
- Translation_Chinese_Dataset.parquet
- Translation_English_Dataset.parquet
- Translation_French_Dataset.parquet
- Translation_Korean_Dataset.parquet
- Translation_Dataset.parquet (aggregated for all languages)

3. segment translated Korean and Chinese news into multiple paragraphs [Notebook - Data Preprocessing]

4. Split training set and validation set [Notebook - Data Preprocessing]
- training_set.parquet
- validation_set.parquet

5. Data augmentation using OpenAI api