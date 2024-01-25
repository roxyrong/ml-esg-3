Original Files:
- ML-ESG-3_Trainset_Chinese.json
- ML-ESG-3_Trainset_English.json
- ML-ESG-3_Trainset_French.json
- ML-ESG-3_Trainset_Japanese.json
- ML-ESG-3_Trainset_Korean.json

Derived Dataset Pipeline: 
1. aggregate all JSON files
- train_df.parquet

2. use DeepL translate non-English news to English and aggregate
- train_df_Chinese_translated.parquet
- train_df_English_translated.parquet
- train_df_French_translated.parquet
- train_df_Korean_translated.parquet
- train_df_all_english.parquet (aggregated for all languages)

3. segment translated Korean and Chinese news into multiple paragraphs
- train_df_Korean_translated_segmented
- train_df_Chinese_translated_segmented

4. augmented training data with GPT4
- train_df_English_augmented.parquet

5. aggregate segmented data and augmented data and final data processing
- training_dataset.parquet
