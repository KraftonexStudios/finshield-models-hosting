import os
import logging
import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import json
import pickle
import yaml

logger = logging.getLogger(__name__)

class ModelNotLoadedException(Exception):
    """Exception raised when trying to use a model that hasn't been loaded"""
    pass

class InvalidInputException(Exception):
    """Exception raised when input data is invalid"""
    pass

class EncoderService:
    """Unified service for managing all three encoder models"""
    
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.model_info = {}
        self.device_map = {}
        
        # Initialize model placeholders
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all enabled models"""
        model_types = ['motion_encoder', 'touch_encoder', 'typing_encoder']
        
        for model_type in model_types:
            if self.config.is_model_enabled(model_type):
                try:
                    self._load_model(model_type)
                    logger.info(f"{model_type} loaded successfully")
                except Exception as e:
                    logger.error(f"Failed to load {model_type}: {str(e)}")
                    # Set placeholder for missing models
                    self._set_model_placeholder(model_type)
    
    def _load_model(self, model_type: str):
        """Load a specific model based on type"""
        model_config = self.config.get_model_config(model_type)
        device = self.config.get_device_config(model_type)
        
        if model_type == 'motion_encoder':
            self._load_motion_encoder(model_config, device)
        elif model_type == 'touch_encoder':
            self._load_touch_encoder(model_config, device)
        elif model_type == 'typing_encoder':
            self._load_typing_encoder(model_config, device)
        
        self.device_map[model_type] = device
    
    def _load_motion_encoder(self, model_config: Dict, device: str):
        """Load motion encoder model"""
        model_path = model_config.get('model_path')
        processor_path = model_config.get('processor_path')
        
        # Check if files exist
        if not os.path.exists(model_path):
            logger.warning(f"Motion encoder model not found at {model_path}, using placeholder")
            self._set_model_placeholder('motion_encoder')
            return
        
        try:
            # Placeholder implementation - will be replaced with actual model loading
            # from motion_encoder.inference import IMUMotionEncoder
            # self.models['motion_encoder'] = IMUMotionEncoder(model_path, processor_path, device)
            
            # For now, create a placeholder
            self.models['motion_encoder'] = MotionEncoderPlaceholder(model_path, processor_path, device)
            self.model_info['motion_encoder'] = {
                'type': 'IMU Motion Encoder',
                'input_features': 11,
                'output_dim': 256,
                'model_path': model_path,
                'processor_path': processor_path,
                'device': device,
                'loaded_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error loading motion encoder: {str(e)}")
            self._set_model_placeholder('motion_encoder')
    
    def _load_touch_encoder(self, model_config: Dict, device: str):
        """Load touch/gesture encoder model"""
        model_path = model_config.get('model_path')
        
        if not os.path.exists(model_path):
            logger.warning(f"Touch encoder model not found at {model_path}, using placeholder")
            self._set_model_placeholder('touch_encoder')
            return
        
        try:
            # Placeholder implementation - will be replaced with actual model loading
            # from touch_encoder.gesture_inference import GestureInference
            # self.models['touch_encoder'] = GestureInference(model_path, device)
            
            # For now, create a placeholder
            self.models['touch_encoder'] = TouchEncoderPlaceholder(model_path, device)
            self.model_info['touch_encoder'] = {
                'type': 'Touch/Gesture Encoder',
                'input_features': 7,
                'output_dim': 256,
                'model_path': model_path,
                'device': device,
                'loaded_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error loading touch encoder: {str(e)}")
            self._set_model_placeholder('touch_encoder')
    
    def _load_typing_encoder(self, model_config: Dict, device: str):
        """Load typing/keystroke encoder model"""
        model_path = model_config.get('model_path')
        metadata_path = model_config.get('metadata_path')
        config_path = model_config.get('config_path')
        
        if not os.path.exists(model_path):
            logger.warning(f"Typing encoder model not found at {model_path}, using placeholder")
            self._set_model_placeholder('typing_encoder')
            return
        
        try:
            # Placeholder implementation - will be replaced with actual model loading
            # from typing_encoder.inference import KeystrokeEncoder
            # self.models['typing_encoder'] = KeystrokeEncoder(model_path, metadata_path, config_path, device)
            
            # For now, create a placeholder
            self.models['typing_encoder'] = TypingEncoderPlaceholder(model_path, metadata_path, config_path, device)
            self.model_info['typing_encoder'] = {
                'type': 'Keystroke Dynamics Encoder',
                'input_features': 'variable (character + 4 numerical)',
                'output_dim': 256,
                'model_path': model_path,
                'metadata_path': metadata_path,
                'config_path': config_path,
                'device': device,
                'loaded_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error loading typing encoder: {str(e)}")
            self._set_model_placeholder('typing_encoder')
    
    def _set_model_placeholder(self, model_type: str):
        """Set a placeholder for a model that couldn't be loaded"""
        self.models[model_type] = None
        self.model_info[model_type] = {
            'type': f'{model_type.replace("_", " ").title()}',
            'status': 'placeholder',
            'message': 'Model file not found - using placeholder',
            'loaded_at': datetime.now().isoformat()
        }
    
    def encode_motion(self, data: Union[Dict, List, np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Encode motion/IMU data"""
        if 'motion_encoder' not in self.models or self.models['motion_encoder'] is None:
            raise ModelNotLoadedException("Motion encoder not loaded")
        
        try:
            return self.models['motion_encoder'].encode_motion(data)
        except Exception as e:
            logger.error(f"Error encoding motion data: {str(e)}")
            raise InvalidInputException(f"Failed to encode motion data: {str(e)}")
    
    def encode_motion_batch(self, data_list: List[Union[Dict, List, np.ndarray, pd.DataFrame]]) -> List[np.ndarray]:
        """Encode batch of motion/IMU data"""
        if 'motion_encoder' not in self.models or self.models['motion_encoder'] is None:
            raise ModelNotLoadedException("Motion encoder not loaded")
        
        try:
            return self.models['motion_encoder'].encode_batch(data_list)
        except Exception as e:
            logger.error(f"Error encoding motion batch: {str(e)}")
            raise InvalidInputException(f"Failed to encode motion batch: {str(e)}")
    
    def encode_gesture(self, data: Union[Dict, List, str]) -> np.ndarray:
        """Encode touch/gesture data"""
        if 'touch_encoder' not in self.models or self.models['touch_encoder'] is None:
            raise ModelNotLoadedException("Touch encoder not loaded")
        
        try:
            return self.models['touch_encoder'].encode_gesture(data)
        except Exception as e:
            logger.error(f"Error encoding gesture data: {str(e)}")
            raise InvalidInputException(f"Failed to encode gesture data: {str(e)}")
    
    def encode_gesture_batch(self, data_list: List[Union[Dict, List, str]]) -> List[np.ndarray]:
        """Encode batch of touch/gesture data"""
        if 'touch_encoder' not in self.models or self.models['touch_encoder'] is None:
            raise ModelNotLoadedException("Touch encoder not loaded")
        
        try:
            return self.models['touch_encoder'].encode_batch(data_list)
        except Exception as e:
            logger.error(f"Error encoding gesture batch: {str(e)}")
            raise InvalidInputException(f"Failed to encode gesture batch: {str(e)}")
    
    def encode_typing(self, data: Union[Dict, List, str, pd.DataFrame]) -> np.ndarray:
        """Encode typing/keystroke data"""
        if 'typing_encoder' not in self.models or self.models['typing_encoder'] is None:
            raise ModelNotLoadedException("Typing encoder not loaded")
        
        try:
            return self.models['typing_encoder'].encode_sequence(data)
        except Exception as e:
            logger.error(f"Error encoding typing data: {str(e)}")
            raise InvalidInputException(f"Failed to encode typing data: {str(e)}")
    
    def encode_typing_batch(self, data_list: List[Union[Dict, List, str, pd.DataFrame]]) -> List[np.ndarray]:
        """Encode batch of typing/keystroke data"""
        if 'typing_encoder' not in self.models or self.models['typing_encoder'] is None:
            raise ModelNotLoadedException("Typing encoder not loaded")
        
        try:
            return self.models['typing_encoder'].encode_sequences(data_list)
        except Exception as e:
            logger.error(f"Error encoding typing batch: {str(e)}")
            raise InvalidInputException(f"Failed to encode typing batch: {str(e)}")
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get status of all models"""
        status = {
            'service_status': 'running',
            'models': {},
            'timestamp': datetime.now().isoformat()
        }
        
        for model_type, model in self.models.items():
            status['models'][model_type] = {
                'loaded': model is not None,
                'info': self.model_info.get(model_type, {}),
                'device': self.device_map.get(model_type, 'unknown')
            }
        
        return status
    
    def get_model_info(self, model_type: str) -> Dict[str, Any]:
        """Get information about a specific model"""
        if model_type not in self.models:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return {
            'model_type': model_type,
            'loaded': self.models[model_type] is not None,
            'info': self.model_info.get(model_type, {}),
            'device': self.device_map.get(model_type, 'unknown')
        }
    
    def reload_model(self, model_type: str):
        """Reload a specific model"""
        if model_type not in ['motion_encoder', 'touch_encoder', 'typing_encoder']:
            raise ValueError(f"Invalid model type: {model_type}")
        
        logger.info(f"Reloading {model_type}")
        
        # Clear existing model
        if model_type in self.models:
            del self.models[model_type]
        if model_type in self.model_info:
            del self.model_info[model_type]
        if model_type in self.device_map:
            del self.device_map[model_type]
        
        # Reload model
        if self.config.is_model_enabled(model_type):
            self._load_model(model_type)
    
    def is_healthy(self) -> bool:
        """Check if the service is healthy"""
        enabled_models = [model_type for model_type in ['motion_encoder', 'touch_encoder', 'typing_encoder'] 
                         if self.config.is_model_enabled(model_type)]
        
        if not enabled_models:
            return False
        
        # Check if at least one model is loaded
        loaded_models = [model_type for model_type in enabled_models 
                        if model_type in self.models and self.models[model_type] is not None]
        
        return len(loaded_models) > 0


# Placeholder classes for when actual models are not available
class MotionEncoderPlaceholder:
    """Placeholder for motion encoder"""
    
    def __init__(self, model_path: str, processor_path: str, device: str):
        self.model_path = model_path
        self.processor_path = processor_path
        self.device = device
        logger.info(f"Motion encoder placeholder initialized (device: {device})")
    
    def encode_motion(self, data) -> np.ndarray:
        """Return dummy embedding for motion data"""
        logger.warning("Using placeholder motion encoder - returning dummy embedding")
        return np.random.randn(256).astype(np.float32)
    
    def encode_batch(self, data_list) -> List[np.ndarray]:
        """Return dummy embeddings for batch motion data"""
        logger.warning("Using placeholder motion encoder - returning dummy embeddings")
        return [np.random.randn(256).astype(np.float32) for _ in data_list]


class TouchEncoderPlaceholder:
    """Placeholder for touch encoder"""
    
    def __init__(self, model_path: str, device: str):
        self.model_path = model_path
        self.device = device
        logger.info(f"Touch encoder placeholder initialized (device: {device})")
    
    def encode_gesture(self, data) -> np.ndarray:
        """Return dummy embedding for gesture data"""
        logger.warning("Using placeholder touch encoder - returning dummy embedding")
        return np.random.randn(256).astype(np.float32)
    
    def encode_batch(self, data_list) -> List[np.ndarray]:
        """Return dummy embeddings for batch gesture data"""
        logger.warning("Using placeholder touch encoder - returning dummy embeddings")
        return [np.random.randn(256).astype(np.float32) for _ in data_list]


class TypingEncoderPlaceholder:
    """Placeholder for typing encoder"""
    
    def __init__(self, model_path: str, metadata_path: str, config_path: str, device: str):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self.config_path = config_path
        self.device = device
        logger.info(f"Typing encoder placeholder initialized (device: {device})")
    
    def encode_sequence(self, data) -> np.ndarray:
        """Return dummy embedding for typing data"""
        logger.warning("Using placeholder typing encoder - returning dummy embedding")
        return np.random.randn(256).astype(np.float32)
    
    def encode_sequences(self, data_list) -> List[np.ndarray]:
        """Return dummy embeddings for batch typing data"""
        logger.warning("Using placeholder typing encoder - returning dummy embeddings")
        return [np.random.randn(256).astype(np.float32) for _ in data_list]