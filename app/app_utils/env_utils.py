import os

def get_env_path(root_dir, var_name, default=None):
    """Retorna um Path do ROOT_DIR + valor da env, garantindo que não seja None ou vazio."""
    value = os.getenv(var_name, default)
    if not value:  # None ou string vazia
        if default is None:
            raise ValueError(f"A variável de ambiente {var_name} não está definida!")
        value = default
    return root_dir / value

def get_env_bool(var_name, default=False):
    """Converte string da env em bool."""
    value = os.getenv(var_name)
    if value is None:
        return default
    return value.strip().lower() in ("true", "1", "yes")

def get_env_int(var_name, default=0):
    """Converte string da env em int."""
    value = os.getenv(var_name)
    if value is None or value.strip() == "":
        return default
    return int(value)