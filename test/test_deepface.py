from deepface import DeepFace

# DeepFaceのバージョン確認
print("DeepFace version:", DeepFace.__version__)

try:
    # 感情分析を実行
    result = DeepFace.analyze("KA.SA1.33.jpg", actions=["emotion"])
    print("Detected Emotion:", result[0]["dominant_emotion"])
except Exception as e:
    print("感情分析に失敗しました:", e)