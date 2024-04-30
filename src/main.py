from models.hermes13B import Hermes13B


if __name__ == "__main__":

    model = Hermes13B()

    prompt = [{"role": "user", "content": "Matthew is excellent and experienced.	Matthew is excellent. a: entailment, b: contradiction or c: neutral?"}]
    output = model.inference_for_prompt(prompt=prompt)

    print("---------------------------")
    print(f"Model prompt: {prompt}")
    print("---------------------------")
    print(f"Model output: {output}")
    print("---------------------------")