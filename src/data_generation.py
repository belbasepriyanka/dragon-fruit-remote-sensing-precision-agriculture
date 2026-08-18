"""Generate deterministic synthetic demo data for the public portfolio."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

SPECIES_EFFECT = {"Red": 1.15, "White": 1.00, "Yellow": 0.82}
ENV_EFFECT = {"High Tunnel": 1.05, "Open Field": 1.00}

def _treatment_response(rate: float) -> float:
    return 1.0 - 0.0025 * (rate - 10.0) ** 2

def generate_demo_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for env in ["High Tunnel", "Open Field"]:
        for species in ["Red", "White", "Yellow"]:
            for treatment in [0, 5, 10, 20]:
                for rep in range(1, 4):
                    plant_id = f"{env[:2].upper()}-{species[0]}-T{treatment}-R{rep}"
                    plant_random = rng.normal(0, 0.08)
                    for dap in [120, 365, 649, 960]:
                        season = dap / 960
                        temp_c = 25.0 + 4.5*np.sin(season*np.pi) + (1.2 if env == "High Tunnel" else 0) + rng.normal(0,0.8)
                        rainfall_mm = max(5, 170 - 90*np.sin(season*np.pi) + (25 if env == "Open Field" else -15) + rng.normal(0,18))
                        treat_eff = _treatment_response(treatment)
                        sp_eff = SPECIES_EFFECT[species]
                        env_eff = ENV_EFFECT[env]
                        soil_n = 16 + treatment*0.55 - 0.0025*dap + (2 if env == "High Tunnel" else 0) + rng.normal(0,1.8)
                        soil_p = 10 + treatment*0.65 + 4*season + rng.normal(0,1.4)
                        soil_k = 105 + treatment*2.2 - (34 if env == "Open Field" and dap >= 649 else 0) + rng.normal(0,9)
                        tissue_n = 1.15 + 0.014*treatment + 0.12*env_eff - 0.18*season + rng.normal(0,0.08)
                        tissue_p = 0.18 + 0.003*treatment + 0.04*season + rng.normal(0,0.015)
                        tissue_k = 1.55 + 0.018*treatment - (0.32 if env == "Open Field" and dap >= 649 else 0) + rng.normal(0,0.11)
                        nutrient_balance = 0.45*tissue_n + 0.35*tissue_k + 0.015*soil_p + 0.002*soil_k
                        heat_penalty = max(temp_c-30,0)*0.06
                        water_penalty = max(90-rainfall_mm,0)*0.003
                        latent_vigor = 0.45 + 0.45*sp_eff + 0.25*env_eff + 0.30*treat_eff + 0.30*nutrient_balance - heat_penalty - water_penalty + plant_random + rng.normal(0,0.06)
                        ndvi = np.clip(0.34 + 0.16*latent_vigor + rng.normal(0,0.025),0.25,0.90)
                        ndre = np.clip(0.16 + 0.12*latent_vigor + 0.10*tissue_n + rng.normal(0,0.02),0.08,0.65)
                        gndvi = np.clip(0.28 + 0.14*latent_vigor + 0.05*tissue_n + rng.normal(0,0.025),0.18,0.82)
                        red_edge_slope = 0.008 + 0.010*latent_vigor + 0.004*tissue_n + rng.normal(0,0.0015)
                        nir_mean = 0.36 + 0.18*latent_vigor + rng.normal(0,0.025)
                        swir_mean = 0.30 - 0.07*latent_vigor + 0.0007*max(90-rainfall_mm,0) + rng.normal(0,0.018)
                        stress_score = 1.30 - 0.50*tissue_n - 0.34*tissue_k - 1.20*ndre + 0.90*swir_mean + 0.09*max(temp_c-29,0) + 0.004*max(90-rainfall_mm,0) + rng.normal(0,0.08)
                        stress_flag = int(stress_score > 0)
                        growth_cm = max(5,32*season*sp_eff*env_eff*treat_eff + 8*latent_vigor + rng.normal(0,3))
                        stem_diameter_mm = max(8,14 + 12*season*sp_eff*treat_eff + rng.normal(0,1.5))
                        if dap < 365:
                            flower_lambda = 0.05
                        else:
                            base = {365:0.6,649:1.8,960:2.4}[dap]
                            flower_lambda = max(0.05,base*sp_eff*treat_eff*(1.08 if env == "Open Field" else 0.92)*max(0.35,1-0.45*stress_flag))
                        flower_count = int(rng.poisson(flower_lambda))
                        fruit_prob = np.clip(0.12 + 0.13*sp_eff + 0.10*treat_eff - 0.13*stress_flag,0.04,0.55)
                        fruit_count = int(rng.binomial(flower_count,fruit_prob)) if flower_count > 0 else 0
                        rows.append({"plant_id":plant_id,"environment":env,"species":species,"treatment_t_acre":treatment,"replicate":rep,"dap":dap,"temperature_c":round(float(temp_c),2),"rainfall_mm":round(float(rainfall_mm),2),"growth_cm":round(float(growth_cm),2),"stem_diameter_mm":round(float(stem_diameter_mm),2),"flower_count":flower_count,"fruit_count":fruit_count,"soil_n_mgkg":round(float(soil_n),2),"soil_p_mgkg":round(float(soil_p),2),"soil_k_mgkg":round(float(soil_k),2),"tissue_n_pct":round(float(tissue_n),3),"tissue_p_pct":round(float(tissue_p),3),"tissue_k_pct":round(float(tissue_k),3),"ndvi":round(float(ndvi),4),"ndre":round(float(ndre),4),"gndvi":round(float(gndvi),4),"red_edge_slope":round(float(red_edge_slope),5),"nir_mean":round(float(nir_mean),4),"swir_mean":round(float(swir_mean),4),"stress_flag":stress_flag})
    return pd.DataFrame(rows)

def save_demo_data(path: str | Path, seed: int = 42) -> Path:
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    generate_demo_data(seed).to_csv(path,index=False)
    return path

if __name__ == "__main__":
    target = Path("data/sample_dragon_fruit_demo.csv")
    save_demo_data(target)
    print(f"Wrote {target}")
