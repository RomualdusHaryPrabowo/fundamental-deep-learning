import pandas as pd #import pandas library to read the csv file
import tensorflow as tf #import tensorflow library to create the windowed dataset

#load the dataset using pandas read_csv function and store it in a variable called data
data = pd.read_csv('dataset/DailyDelhiClimateTrain.csv')
#display the first 5 rows of the dataset to understand its structure and contents
print(data.head())

# define a function called windowed_dataset that takes in a time series, window size, batch size, and shuffle buffer as parameters
# series: the input time series data
# window_size: the size of the window
# batch_size: the size of the batches to be created
# shuffle_buffer: the buffer size for shuffling the dataset (Pengacakan dataset)
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
 series = tf.expand_dims(series, axis=-1) # mengubah dimensi dari series menjadi 2D dengan menambahkan dimensi baru di akhir    
 ds = tf.data.Dataset.from_tensor_slices(series) # membuat dataset dari tensor series dengan menggunakan tf.data.Dataset.from_tensor_slices
 ds = ds.window(window_size + 1, shift=1, drop_remainder=True) # membuat window pada dataset dengan ukuran window_size + 1, pergeseran sebesar 1, dan drop_remainder=True untuk memastikan bahwa semua window memiliki ukuran yang sama
 ds = ds.flat_map(lambda w: w.batch(window_size + 1)) # mengubah dataset menjadi batch dengan ukuran window_size + 1 menggunakan flat_map dan batch
 ds = ds.shuffle(shuffle_buffer) # mengacak dataset dengan menggunakan shuffle_buffer sebagai ukuran buffer untuk pengacakan
 ds = ds.map(lambda w: (w[:-1], w[-1:])) # memetakan dataset untuk memisahkan input (w[:-1]) dan target (w[-1:]) dari setiap window (Memasukkan semua elemen kecuali yang terakhir sebagai input dan elemen terakhir sebagai target)
 return ds.batch(batch_size).prefetch(1) # mengembalikan dataset yang sudah dibatch dengan ukuran batch_size dan menggunakan prefetch untuk meningkatkan performa saat pelatihan model

# menentukan fitur yang akan digunakan 
dates = data['date'].values #mengambil kolom 'date' dari dataset dan menyimpannya dalam variabel dates
temperatures = data['meantemp'].values #mengambil kolom 'meantemp' dari dataset dan menyimpannya dalam variabel temperatures
print (dates[:5]) #menampilkan 5 nilai pertama dari variabel dates untuk memeriksa isinya
print (temperatures[:5]) #menampilkan 5 nilai pertama dari variabel temperatures untuk memeriksa isinya

# call the windowed_dataset function 
train_set = windowed_dataset(temperatures, window_size=2, batch_size=3, shuffle_buffer=1000)

# looping for print data
for data in train_set:
 print(data)
 break #menghentikan loop setelah mencetak data pertama untuk menghindari pencetakan seluruh dataset