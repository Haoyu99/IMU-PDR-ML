import data_ridi
import matplotlib.pyplot as plt

data = data_ridi.RIDIRawDataSequence('D:\DataSet\RIDI\\archive\data_publish_v2\dan_bag1\processed',interval=200)
print(data.get_aux().shape)