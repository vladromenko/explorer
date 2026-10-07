"""Read independent JSON records without letting one damaged file break the UI."""
import json
import math
from pathlib import Path


def records(paths,required=()):
    valid=[];errors=[]
    for path in paths:
        try:
            value=json.loads(Path(path).read_text())
            if not isinstance(value,dict) or any(key not in value for key in required):
                raise ValueError("Missing record fields: "+", ".join(required))
            for key in required:
                field=value[key]
                if key in ("id","name","state","outcome") and not isinstance(field,str):
                    raise ValueError("Invalid string field: "+key)
                if key in ("started","at") and (type(field) not in (int,float) or not math.isfinite(field)):
                    raise ValueError("Invalid timestamp field: "+key)
                if key=="steps" and not isinstance(field,list):raise ValueError("Invalid episode steps")
            valid.append((Path(path),value))
        except (OSError,ValueError,TypeError) as exc:
            errors.append({"file":str(path),"error":str(exc)})
    return valid,errors
