from src.config_manager import get_config
from src.model_loader import load_model
from src.api_server import APIServer
from src.logger import configure_logging, get_logger

def main():
 
    config = get_config()
    configure_logging(config.get("log_level"))
    logger = get_logger(__name__)
    
   
    logger.info("Starting ML serving engine")
    logger.info(f"Using model: {config['model_path']}")
    
    try: 
        model = load_model(
            model_path=config["model_path"],
            format=config.get("model_format")
        )   
        server = APIServer(
            model=model,
            host=config["host"],
            port=config["port"]
        ) 
        server.run()
        
    except Exception as e:
        logger.error(f"Error running ML serving engine: {str(e)}")
        raise

if __name__ == "__main__":
    main()