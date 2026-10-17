run generate_data.py to get dataset, it names X.npy and y.npy
run train_e.py to get mlp_weights.json ,it is the MLP weight. Than put mlp_weights.json in javascripts.
In the basefile, run a httpserver for example use python :  python -m http.server 8000 .
open localhost:8000/indexe.html with your web.
