class TemplateFiller:
    def __init__(self, template_path: str, values: dict):
        self.template_path = template_path
        self.values = values
    
    def save(self, output_path: str):
        with open(self.template_path) as f: 
            template = f.read()

        filled_template = template.format(**self.values)
        with open(output_path, 'w') as f:
            f.write(filled_template)



class InitTemplate(TemplateFiller):
    def __init__(self, emiss: float, h2o: float, t_end: int, dt_save: int):
        values = {
            "EMISS": emiss,
            "H20": round(h2o, 4),
            "TEND": int(t_end),
            "DTSAVE": int(dt_save)
        }

        super().__init__("templates/init_template.txt", values)



class VentTemplate(TemplateFiller): 
    def __init__(self, lon: int, lat: int, t_start: int, t_end: int, flux_pct: float, fluxrate: str):
        values = {
            "LON": int(lon),
            "LAT": int(lat),
            "TSTART": int(t_start),
            "TEND": int(t_end),
            "FLUX_PCT": round(flux_pct, 4),
            "FLUXRATE": fluxrate
        }

        super().__init__("templates/vent_template.txt", values)



if __name__ == "__main__":
    pass

