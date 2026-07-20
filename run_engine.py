from src.config_manager import get_config
from src.model_loader import load_model
from src.api_server import APIServer
from src.logger import configure_logging, get_logger
from src.storage import create_artifact_storage

def main():
 
    config = get_config()
    configure_logging(config.get("log_level"))
    logger = get_logger(__name__)
    
   
    logger.info("Starting ML serving engine")
    logger.info(f"Using model: {config['model_path']}")
    storage = create_artifact_storage(config)
    
    try:
        model = load_model(
            model_path=config["model_path"],
            format=config.get("model_format"),
            quantization=config.get("quantization"),
            adapter_path=config.get("adapter_path"),
            merge_adapter=bool(config.get("merge_adapter", False)),
            trusted_model_paths=config.get("trusted_model_paths", []),
            allow_unsafe_deserialization=bool(config.get("allow_unsafe_deserialization", False)),
            storage=storage,
        )
        server = APIServer(
            model=model,
            host=config["host"],
            port=config["port"],
            config=config,
        ) 
        server.run()
        
    except FileNotFoundError as e:
        if not config.get("allow_empty_model", False):
            raise
        logger.warning(f"Starting without a loaded model: {e}")
        APIServer(
            model=None,
            host=config["host"],
            port=config["port"],
            config=config,
        ).run()
    except Exception as e:
        logger.error(f"Error running ML serving engine: {str(e)}")
        raise

if __name__ == "__main__":
    main()
