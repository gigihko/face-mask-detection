# face-mask-detection

jadi bgni mas hehe,
project ini dasarnya python

source code dasarnya dah ada di folder jupyter,
nah dibikin user interface pake framework Streamlit

--------------------------------------------------------------

untuk objektifnya nanti ada beberapa fitur 

fitur pertama itu buat nampilin datasetnya 
nah disini datasetnya ada 2 
yang ditampilin nanti cuma face mask datasetnya 

ini linknya https://www.kaggle.com/datasets/ashishjangra27/face-mask-12k-images-dataset

nanti ditampilin total file dlm datasetnya
& sampel random aja kyk yg di jupyter notebook

--------------------------------------------------------------

selanjutnya fitur utama 
Deteksi wajah manusia pake metode Haar Cascade

outputnya 3 yaitu nampilin gambar originalnya 
terus di proses pake open cv jadi terdeteksi wajahnya 
file haar cascadenya tinggal dipake yang frontal face.xml ada di dlm folder Haar Cascade https://github.com/gigihko/face-mask-detection/blob/main/haarcascades/haarcascade_frontalface_default.xml


selanjutnya fitur paling utamanya 
deteksi wajah juga cuma ditambahin model VGG-19

ini modelnya sudah di training tinggal dipakai https://drive.google.com/file/d/1gHSkEZxN-VdixuyrK-To13R3xo5-BB2L/view?usp=sharing

bertujuan uttuk deteksi wajah dengan akurasi lebih tinggi 
karena sdh menggunakan training data pengguna masker 

outputnya juga sama 3 yaitu nampilin gambar original
terus di proses pake opencv dgn tambahan model VGG-19
jadi terdeteksi wajah manusianya dan bisa dibedakan dia menggunakan masker atau tidak

--------------------------------------------------------------

nah yang terakhir data akurasinya 
nanti ada 3 
data akurasi menggunakan haar cascade itu bisa mndeteksi berapa wajah 
yang dipakai nanti cuma folder data test dari dataset face mask detection

terus data akurasi training model VGG-19
terakhir data akurasi setelah digabungin bisa terdeteksi berapa pengguna masker dan berapa yang tdk menggunakan masker

dah itu mas :)
thanks
