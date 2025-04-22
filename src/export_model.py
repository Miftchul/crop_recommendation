import pickle

# Simpan model dengan pickle
with open('crop_recommendation_pipeline.pkl', 'wb') as file:
    pickle.dump(pipeline, file)

print("Model berhasil diekspor menggunakan pickle.")
