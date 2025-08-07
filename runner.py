from transformer_trainer import translate

def main():
    test_sentences = [
        "i will",
        "hello world", 
        "how are you",
        "i am fine"
    ]
    
    print("🤖 Testing Translation Model")
    print("=" * 40)
    
    for text in test_sentences:
        try:
            print(f"\n📤 Translating: '{text}'")
            
            output_greedy = translate(text, temperature=0.0)
            print(f"Greedy:     {output_greedy}")
            
            output_sample = translate(text, temperature=0.8)
            print(f"Sampling:   {output_sample}")
            
            output_creative = translate(text, temperature=1.2)  
            print(f"Creative:   {output_creative}")
            
        except Exception as e:
            print(f"\n❌ Error translating '{text}': {e}")

if __name__ == "__main__":
    main()