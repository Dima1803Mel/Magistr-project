"""
Единый интерфейс для всех моделей.
Позволяет FastAPI вызывать model.predict(X) независимо от библиотеки.
"""

from abc import ABC, abstractmethod
import joblib
import numpy as np
import pandas as pd


# ============================================================
# 1. Абстрактный класс (интерфейс)
# ============================================================
class BaseModel(ABC):
    """Базовый класс для всех моделей."""

    def __init__(self, name: str):
        self.name = name
        self.model = None

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def save(self, path: str):
        joblib.dump(self.model, path)

    @classmethod
    def load(cls, path: str, name: str = None):
        instance = cls(name or path)
        instance.model = joblib.load(path)
        return instance


# ============================================================
# 2. Реализации для каждой библиотеки
# ============================================================
class SklearnModel(BaseModel):
    """Для LinearRegression, DecisionTree, RandomForest, SVM, k-NN."""

    def __init__(self, name: str, model=None):
        super().__init__(name)
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class XGBoostModel(BaseModel):
    """Для XGBoost."""

    def __init__(self, name: str, model=None):
        super().__init__(name)
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class CatBoostModel(BaseModel):
    """Для CatBoost."""

    def __init__(self, name: str, model=None):
        super().__init__(name)
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class LightGBMModel(BaseModel):
    """Для LightGBM."""

    def __init__(self, name: str, model=None):
        super().__init__(name)
        self.model = model

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


class FastAIModel(BaseModel):
    """Для MLP на FastAI. FastAI learner.predict возвращает tuple."""

    def __init__(self, name: str, model=None):
        super().__init__(name)
        self.model = model

    def fit(self, X, y):
        raise NotImplementedError("FastAI обучается через Learner.fit(), не через .fit(X, y)")

    def predict(self, X):
        # FastAI learner.predict возвращает (pred, tensor, probs) — берём [0]
        if isinstance(X, pd.DataFrame):
            preds = [self.model.predict(row)[0] for _, row in X.iterrows()]
            return np.array(preds)
        return np.array([self.model.predict(x)[0] for x in X])


# ============================================================
# 3. Фабрика: создаёт модель по имени
# ============================================================
MODEL_REGISTRY = {
    "linear": SklearnModel,
    "poly": SklearnModel,
    "tree": SklearnModel,
    "forest": SklearnModel,
    "svm": SklearnModel,
    "knn": SklearnModel,
    "xgboost": XGBoostModel,
    "catboost": CatBoostModel,
    "lightgbm": LightGBMModel,
    "mlp": FastAIModel,
}


def create_model(name: str, model=None):
    """Создаёт обёртку для модели по её имени."""
    key = name.lower().replace(" ", "_")
    if key not in MODEL_REGISTRY:
        raise ValueError(f"Неизвестная модель: {name}. Доступны: {list(MODEL_REGISTRY.keys())}")
    return MODEL_REGISTRY[key](name=name, model=model)


# ============================================================
# 4. Хранилище всех моделей (используется в FastAPI)
# ============================================================
class ModelStore:
    """Загружает все .pkl из папки и предоставляет единый predict."""

    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        self.models: dict[str, BaseModel] = {}

    def load_all(self):
        import os
        if not os.path.exists(self.models_dir):
            print(f"⚠️ Папка {self.models_dir} не найдена")
            return self

        for fname in os.listdir(self.models_dir):
            if not fname.endswith(".pkl"):
                continue
            name = fname.replace(".pkl", "")
            path = os.path.join(self.models_dir, fname)
            try:
                raw = joblib.load(path)
                wrapper = create_model(name, model=raw)
                self.models[name] = wrapper
                print(f"✅ Загружена: {name}")
            except Exception as e:
                print(f"❌ Ошибка загрузки {name}: {e}")
        return self

    def predict(self, model_name: str, X):
        """Предсказание одной моделью."""
        if model_name not in self.models:
            raise KeyError(f"Модель {model_name} не загружена")
        return self.models[model_name].predict(X)

    def predict_all(self, X) -> dict:
        """Предсказание всеми моделями сразу."""
        results = {}
        for name, wrapper in self.models.items():
            try:
                results[name] = float(wrapper.predict(X)[0])
            except Exception as e:
                results[name] = f"error: {e}"
        return results

    def list_models(self) -> list[str]:
        return list(self.models.keys())