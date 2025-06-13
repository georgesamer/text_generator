from text_generator import TextGenerator

def main():
    # إنشاء كائن جديد من مولد النصوص
    generator = TextGenerator()

    # اختبار النموذج مع نصوص مختلفة
    generator.test_model("Artificial intelligence will")
    generator.test_model("In the future, humans and machines")
    generator.test_model("Education powered by AI can")

    # تحضير نصوص التدريب
    training_texts = [
        "Artificial intelligence is the future of technology",
        "Machines can learn from their experiences",
        "Deep learning mimics the human brain"
    ]
    
    # تدريب النموذج على النصوص المخصصة
    generator.fine_tune(training_texts, epochs=2)

    # اختبار النموذج بعد التدريب
    print("\nTesting after fine-tuning:")
    generator.test_model("The potential of AI in")

if __name__ == "__main__":
    main()
