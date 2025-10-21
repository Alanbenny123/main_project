# Model Setup Guide 🤖

This guide explains how to train and integrate the behavior detection model.

## Option 1: Train the Model from Scratch

### Step 1: Download Dataset

1. Go to [Kaggle Dataset](https://www.kaggle.com/datasets/phamluhuynhmai/classroom-student-behaviors)
2. Click "Download" (you may need to create a Kaggle account)
3. Extract the downloaded ZIP file
4. You should see a folder named `Behaviors_Features`

### Step 2: Setup Data Directory

Create the data directory and move the dataset:

```bash
# Create data directory
mkdir -p data

# Move the extracted Behaviors_Features folder into data/
# Your structure should be:
# data/
#   └── Behaviors_Features/
#       ├── Raising_Hand/
#       ├── Reading/
#       ├── Sleeping/
#       └── Writting/
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- TensorFlow
- TensorFlow Addons
- OpenCV
- scikit-image
- NumPy, Pandas, scikit-learn
- Matplotlib
- Jupyter

### Step 4: Train the Model

1. Start Jupyter Notebook:
```bash
jupyter notebook
```

2. Open `Student_Behaviors_NoteBook_1 (2).ipynb`

3. Run all cells in order (Runtime > Run all or Shift+Enter through each cell)

4. The training process will:
   - Load and preprocess the data
   - Create train/validation/test splits
   - Build the Swin Transformer model
   - Train for multiple epochs
   - Save the trained model

### Step 5: Save the Model

At the end of the notebook, add this cell to save the model:

```python
# Save the model
model.save('./saved_model/student_behavior_model.h5')
print("Model saved successfully!")
```

Run it, and your trained model will be saved to `saved_model/student_behavior_model.h5`

### Step 6: Update Backend

The backend (`app.py`) automatically looks for the model at:
```python
model_path = './saved_model/student_behavior_model.h5'
```

Restart the Flask server, and it will load your trained model!

---

## Option 2: Use a Pre-trained Model

If you have a pre-trained model file:

1. Create the model directory:
```bash
mkdir saved_model
```

2. Place your model file as:
```
saved_model/student_behavior_model.h5
```

3. Start the Flask backend:
```bash
python app.py
```

The app will automatically load the model.

---

## Model Architecture

The system uses **Swin Transformer** architecture:

- **Input**: 224x224x3 RGB images
- **Architecture**: Swin Transformer (State-of-the-art Vision Transformer)
- **Output**: 4 classes (Raising Hand, Reading, Sleeping, Writing)
- **Training samples**: 460
- **Validation samples**: 116
- **Test samples**: 144

---

## Verifying Model is Loaded

When you start the backend, you should see:

```
Starting Student Behavior Analysis System...
Initializing database...
Loading model...
✓ Model loaded successfully!
Server ready!
```

In the web UI, the status indicator should show:
- 🟢 **Model Ready** (green) - Model loaded successfully
- 🔴 **Model Not Loaded (Demo Mode)** (red) - Running with mock predictions

---

## Model Performance Tips

### For Better Accuracy:

1. **More Training Data**: Use all available samples from the dataset
2. **Data Augmentation**: Add rotation, flip, brightness adjustments
3. **Longer Training**: Increase epochs (try 50-100)
4. **Fine-tuning**: Adjust learning rate and batch size

### Example Training Code (add to notebook):

```python
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Callbacks
checkpoint = ModelCheckpoint(
    'best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

early_stop = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True,
    verbose=1
)

# Train with callbacks
history = model.fit(
    train_generator,
    epochs=50,
    validation_data=val_generator,
    callbacks=[checkpoint, early_stop]
)
```

---

## Troubleshooting

### Error: "Model file not found"
**Solution**: Make sure the model file exists at `./saved_model/student_behavior_model.h5`

### Error: "Unable to load model"
**Solution**: 
- Check TensorFlow version compatibility
- Try: `pip install tensorflow==2.10.0`
- Rebuild the model if saved with a different TensorFlow version

### Error: "CUDA/GPU errors"
**Solution**:
- For CPU-only: `pip install tensorflow-cpu`
- For GPU: Install CUDA toolkit and cuDNN
- Model works fine on CPU for inference

### Low Accuracy
**Solution**:
- Train for more epochs
- Use data augmentation
- Ensure dataset is properly loaded
- Check class balance

---

## Demo Mode

The app works **without a trained model** in demo mode:
- Uses random predictions for testing
- Perfect for UI development and testing
- Switch to real model when ready

---

## Export/Deploy Model

### For Production:

1. **Convert to TensorFlow Lite** (mobile/edge):
```python
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('model.tflite', 'wb') as f:
    f.write(tflite_model)
```

2. **Convert to ONNX** (cross-platform):
```python
import tf2onnx
onnx_model, _ = tf2onnx.convert.from_keras(model)
with open("model.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())
```

3. **TensorFlow Serving** (cloud deployment):
```bash
tensorflow_model_server \
    --model_base_path=/path/to/saved_model \
    --model_name=student_behavior
```

---

## Next Steps

1. ✅ Train the model with your dataset
2. ✅ Test predictions in the web UI
3. ✅ Use Live Analysis for real-time detection
4. ✅ Generate reports for students
5. ✅ Monitor and improve model performance

---

## Questions?

- Check the main [README.md](README.md) for general setup
- See the Jupyter notebook for detailed training code
- Model performs best with good lighting and clear student visibility


