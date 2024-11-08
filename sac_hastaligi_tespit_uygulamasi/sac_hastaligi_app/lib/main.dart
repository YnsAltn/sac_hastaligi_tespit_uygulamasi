import 'dart:io';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite/tflite.dart';

void main() {
  runApp(SacHastaligiTeshisApp());
}

class SacHastaligiTeshisApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Saç Hastalığı Teşhis Uygulaması',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: TeshisEkrani(),
    );
  }
}

class TeshisEkrani extends StatefulWidget {
  @override
  _TeshisEkraniState createState() => _TeshisEkraniState();
}

class _TeshisEkraniState extends State<TeshisEkrani> {
  File? _image;
  bool _isLoading = false;
  List? _results;

  @override
  void initState() {
    super.initState();
    loadModel();
  }

  Future<void> loadModel() async {
    String? res = await Tflite.loadModel(
      model: "assets/sac_hastaliklari2_model.tflite",
      labels: "assets/labels.txt",
    );
    print("Model Yüklendi: $res");
  }

  Future<void> selectImage() async {
    final picker = ImagePicker();
    final pickedFile = await picker.pickImage(source: ImageSource.gallery);

    if (pickedFile != null) {
      setState(() {
        _isLoading = true;
        _image = File(pickedFile.path);
      });
      classifyImage(_image!);
    }
  }

  Future<void> classifyImage(File image) async {
    var recognitions = await Tflite.runModelOnImage(
      path: image.path,
      imageMean: 127.5,
      imageStd: 127.5,
      numResults: 1,
      threshold: 0.7,
    );

    setState(() {
      _isLoading = false;
      _results = recognitions;
    });
  }

  @override
  void dispose() {
    Tflite.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text('Saç Hastalığı Teşhis'),
      ),
      body: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            _image == null
                ? Text('Bir resim seçin.')
                : Image.file(_image!),
            SizedBox(height: 16),
            _results != null
                ? Text(
              "Tahmin: ${_results![0]["label"]}\nOlasılık: ${(_results![0]["confidence"] * 100).toStringAsFixed(2)}%",
              style: TextStyle(fontSize: 20),
            )
                : Text(""),
            SizedBox(height: 16),
            ElevatedButton(
              onPressed: selectImage,
              child: Text('Galeriden Resim Seç'),
            ),
          ],
        ),
      ),
    );
  }
}
