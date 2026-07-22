import os

class ModelZoo:
    @classmethod
    def download(cls, model_name, framework="pytorch", output_dir="models"):
        """Download model from supported zoos"""
        if framework == "huggingface":
            return cls._download_huggingface(model_name, output_dir) 
        elif framework == "tensorflow":
            return cls._download_tf_hub(model_name, output_dir)
        else:
            raise ValueError(f"Unsupported framework: {framework}")

    @staticmethod
    def _download_huggingface(model_name, output_dir):
        # Keep this optional utility import-safe when serving non-torch models.
        import torch
        from transformers import AutoModel

        model = AutoModel.from_pretrained(model_name)
        output_path = os.path.join(output_dir, f"{model_name.replace('/', '_')}.pt")
        torch.save(model.state_dict(), output_path)
        return output_path

    @staticmethod
    def _download_tf_hub(model_name, output_dir):
     
        raise NotImplementedError("TensorFlow Hub download not yet implemented")

if __name__ == "__main__":
    model_path = ModelZoo.download("bert-base-uncased", framework="huggingface")
    print(f"Model downloaded to: {model_path}")
