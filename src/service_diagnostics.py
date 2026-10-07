"""Bounded, read-only systemd inspection; an unavailable reply is not 'inactive'."""
import subprocess


def inspect_services(units, runner=subprocess.run):
    try:
        response=runner(["systemctl","--user","is-active",*units.values()],
                        capture_output=True,text=True,timeout=3)
    except (OSError,subprocess.SubprocessError) as exc:
        return {"services":{name:"unknown" for name in units},"services_error":str(exc)}
    values=response.stdout.splitlines()
    services={name:values[index].strip() or "unknown" if index<len(values) else "unknown"
              for index,name in enumerate(units)}
    return {"services":services,"services_error":None if len(values)==len(units) else "Incomplete systemd response"}
