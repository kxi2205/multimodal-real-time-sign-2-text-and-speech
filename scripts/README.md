# Gesture Management Scripts

This directory contains standalone scripts for recording and testing ASL gestures independently from the main web application.

---

## 📁 Scripts Overview

### 1. `record_gestures.py` - Record New Phrases/Words

**Purpose**: Record new gesture templates that will be automatically available in the main application.

**Features**:
- Interactive recording with webcam
- Support for single-hand and two-hand gestures
- Automatic normalization and saving
- Visual feedback during recording
- Saves directly to `backend/templates/` (auto-loaded by backend)

**Usage**:
```bash
cd scripts
py record_gestures.py
```

**Recording Process**:
1. Window opens with webcam feed
2. Enter gesture name (e.g., "thank you", "sorry", "please")
3. Show your sign to the camera
4. Click "Capture" when hand pose is detected
5. Gesture is saved as `.npy` file
6. Repeat for more gestures

**Tips**:
- Use descriptive names: "good morning", "nice to meet you"
- For questions, use "?" in name: "how are you?"
- Practice the sign before recording
- Ensure good lighting and clear hand visibility
- Two-hand gestures are automatically detected

**File Location**:
- Saved to: `backend/templates/{gesture_name}.npy`
- Automatically available after backend restart

---

### 2. `test_recognition.py` - Test Recognition System

**Purpose**: Test the integrated recognition system without running the full web application.

**Features**:
- Tests both alphabet (A-Z) and phrase recognition
- Same recognition logic as the backend
- Real-time visual feedback
- Shows recognition type (letter vs phrase)
- Displays confidence scores

**Usage**:
```bash
cd scripts
py test_recognition.py
```

**What You'll See**:
- **Hands**: Number of detected hands (0-2)
- **Type**: Recognition type (LETTER, PHRASE, or WAITING)
- **Detected**: The recognized text
- **Confidence**: Accuracy percentage

**Color Coding**:
- 🟢 **Green** = Phrase detected
- 🟠 **Orange** = Letter detected
- ⚪ **White** = Waiting for gesture

**Testing Strategy**:
1. Test single letters (A, B, C...) - should show LETTER
2. Test recorded phrases (hello, hi, good...) - should show PHRASE
3. Test new gestures you just recorded
4. Verify confidence scores are reasonable (>50%)

---

## 🔄 Workflow: Adding New Gestures

### Step 1: Record Gesture
```bash
cd scripts
py record_gestures.py
```
- Enter name: "thank you"
- Show the sign
- Click "Capture"
- File saved: `backend/templates/thank_you.npy`

### Step 2: Test Gesture (Optional)
```bash
py test_recognition.py
```
- Sign "thank you"
- Should show: "thank you" (PHRASE, green)
- Verify confidence is good

### Step 3: Restart Backend
```bash
cd ../backend
py app.py
```
- Backend auto-loads new templates
- Look for: "Loaded 10 templates" (now includes your new gesture)

### Step 4: Use in Main App
- Open http://localhost:3000
- Sign "thank you"
- Should display green phrase text
- Automatically spoken!

---

## 📋 Requirements

Both scripts require the same dependencies as the backend:

```bash
pip install -r backend/requirements.txt
```

**Required packages**:
- opencv-python
- mediapipe
- numpy
- joblib (for test script)
- scikit-learn (for test script)

---

## 🎯 Current Templates

After initial setup, you have these phrases:

1. bad
2. fine  
3. good
4. hello
5. hi
6. how you?
7. so so
8. what's up?
9. you good?

Plus: **A-Z alphabet letters** (26 total)

---

## 💡 Recording Tips

### Best Practices:
- ✅ **Good lighting** - Bright, even lighting
- ✅ **Plain background** - Solid color behind you
- ✅ **Clear hands** - All fingers visible
- ✅ **Hold steady** - Keep pose stable when capturing
- ✅ **Natural position** - Sign as you normally would

### Naming Conventions:
- Use lowercase: "good morning" not "Good Morning"
- Use spaces: "thank you" not "thankyou"
- For questions: "how are you?" (? is auto-encoded)
- Avoid special characters: `< > / \ | * :`

### Recording Strategy:
1. **Practice first** - Know the sign well before recording
2. **Multiple attempts** - Record same sign 2-3 times, pick best
3. **Test immediately** - Use test script to verify
4. **Natural speed** - Don't rush, sign naturally

---

## 🔧 Troubleshooting

### "No hands detected"
- Move closer to camera
- Ensure good lighting
- Check camera permissions
- Try different background

### "Template not loading"
- Check file saved to `backend/templates/`
- Restart backend after adding templates
- Verify `.npy` file extension
- Check file isn't corrupted (should be a few KB)

### "Low confidence scores"
- Re-record gesture more clearly
- Ensure consistent hand position
- Check lighting conditions
- Practice sign accuracy

### "Backend import errors in test script"
- Make sure you're in the `scripts/` directory
- Verify `backend/template_matcher.py` exists
- Check Python path is correct
- Install all requirements

---

## 🚀 Advanced Usage

### Recording Two-Hand Gestures:
- Just show both hands in frame
- System auto-detects 2 hands
- Saved as (42,3) shape array
- Works seamlessly with matching

### Batch Recording:
Record multiple gestures in one session:
1. Start `record_gestures.py`
2. Record first gesture
3. Window stays open
4. Record next gesture
5. Continue until done
6. Close window when finished

### Updating Existing Gestures:
1. Delete old `.npy` file from `backend/templates/`
2. Record new version with same name
3. New template replaces old one
4. Restart backend

---

## 📊 File Formats

### Template Files (.npy):
```
backend/templates/
  ├── hello.npy          # Single hand (21,3) or two hands (42,3)
  ├── good.npy           # Normalized landmarks
  ├── thank_you.npy      # Spaces replaced with _
  └── how_are_you_question.npy  # ? becomes _question
```

**File Structure**:
- NumPy array format
- Shape: (21, 3) for 1 hand, (42, 3) for 2 hands
- Values: Normalized landmarks (mean ~0, scale ~1)
- Size: Typically 2-5 KB per file

---

## 🎓 Recognition Algorithm

Both scripts use the same logic as the backend:

1. **MediaPipe Detection**: Extract hand landmarks (21 points × 3 coords)
2. **Normalization**: Translate to origin, scale by max distance
3. **Template Matching**: Weighted Euclidean distance
   - Fingertips: 3× weight
   - Joints: 2× weight
   - Palm: 1× weight
4. **Threshold Check**: Distance < 15.0 for match
5. **Fallback**: Single hand → try alphabet model
6. **Result**: Return best match with confidence

---

## 🌟 Quick Reference

| Task | Command | Output |
|------|---------|--------|
| Record new gesture | `py record_gestures.py` | `.npy` file in `backend/templates/` |
| Test recognition | `py test_recognition.py` | Real-time recognition window |
| List templates | Check `backend/templates/` | All `.npy` files |
| Delete template | Delete `.npy` file | Gone after backend restart |

---

## ✨ Example Session

```bash
# 1. Record a new gesture
cd scripts
py record_gestures.py
# Enter: "good morning"
# Show sign, capture
# Exit window

# 2. Test it works
py test_recognition.py  
# Sign "good morning"
# See: "good morning" (PHRASE, green)
# Press Q

# 3. Use in main app
cd ../backend
py app.py
# See: "Loaded 10 templates: ..., good morning"

# 4. Open frontend
# Go to http://localhost:3000
# Sign "good morning"
# Watch it appear as green phrase text!
```

---

**Remember**: Templates are automatically loaded by the backend on startup. Just record, restart backend, and your new gestures are ready to use! 🎉
