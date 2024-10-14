from transformers import pipeline
checkpoint = "intfloat/multilingual-e5-large"
feature_extractor = pipeline("feature-extraction", framework="pt", model=checkpoint)
text = """COVER AGE AREA  OF XL SATU  
 
Layanan internet rumah XL SATU dari XL Axiata kini telah menjangkau 93 kota di 
seluruh Indonesia. Jangkauan yang luas ini memungkinkan lebih banyak keluarga di 
Indonesia untuk menikmati koneksi internet cepat dan stabil untuk menunjang berbagai 
aktivitas digital mereka.  
Dokumen ini disusun untuk memberikan daftar lengkap kota yang saat ini tercakup 
layanan XL SATU. Daftar ini akan diperbarui secara berkala seiring dengan komitmen XL"""