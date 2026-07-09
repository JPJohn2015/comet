# python imports
import numpy as np

# --------------------------------------------------------------------------------------------------------------------------
class SatelliteProperties:
    """Class that hold Satellite Properties.

    Example Constructions:
        * sp = SatelliteProperties()
        * sp = SatelliteProperties(dry_mass, wet_mass, area, Cd, Cr)
    """
    # ----------------------------------------------------------------------------------------------------------------------
    # Class Construction
    # ----------------------------------------------------------------------------------------------------------------------
    def __init__(self, 
                 dry_mass: int|float = 100.0, 
                 wet_mass: int|float = 0.0, 
                 area: int|float = 1.0, 
                 Cd: int|float = 2.2, 
                 Cr: int|float = 1.5):
        """Defines various Satellite Properties.

        Args:
            dry_mass (int|float, optional): Satellite Dry Mass in kg. Defaults to 100.0.
            wet_mass (int|float, optional): Satellite Wet Mass in kg. Defaults to 0.0.
            area (int|float, optional): Satellite Area in m^2. Defaults to 1.0.
            Cd (int|float, optional): Satellite Coefficient of Drag. Defaults to 2.2.
            Cr (int|float, optional): Satellite Coefficient of Reflectivity. Defaults to 1.5.
        """
        # Assign Class Attributes
        if not isinstance(dry_mass, int|float) or (dry_mass < 0.0):
            raise TypeError('SatelliteProperties(): dry_mass must be an integer or float greater than 0.0')
        self.dry_mass = dry_mass
        if not isinstance(wet_mass, int|float) or (wet_mass < 0.0):
            raise TypeError('SatelliteProperties(): wet_mass must be an integer or float greater than 0.0')
        self.wet_mass = wet_mass
        if not isinstance(area, int|float) or (area < 0.0):
            raise TypeError('SatelliteProperties(): area must be an integer or float greater than 0.0')
        self.area = area
        if not isinstance(Cd, int|float) or (Cd < 0.0):
            raise TypeError('SatelliteProperties(): Cd must be an integer or float greater than 0.0')
        self.Cd = Cd
        if not isinstance(Cr, int|float) or (Cr < 0.0):
            raise TypeError('SatelliteProperties(): Cr must be an integer or float greater than 0.0')
        self.Cr = Cr

    # ----------------------------------------------------------------------------------------------------------------------
    # Class Methods
    # ----------------------------------------------------------------------------------------------------------------------
    def copy(self):
        """Returns a copy of the SatelliteProperties.

        Returns:
            SatelliteProperties: Copy of the SatelliteProperties.
        """
        return SatelliteProperties(self.dry_mass, self.wet_mass, self.area, self.Cd, self.Cr)
    
    # ----------------------------------------------------------------------------------------------------------------------
    def area_to_mass(self) -> float:
        """Returns the Area-to-Mass Ratio.

        Returns:
            area_to_mass (float): satellite Area-to-Mass Ratio in km^2/kg.
        """
        return (self.area*(10e-6))/(self.dry_mass + self.wet_mass)

    # ----------------------------------------------------------------------------------------------------------------------
    def get_dry_mass(self) -> float:
        """Returns the satellite Dry Mass.

        Returns:
            dry_mass (float): satellite Dry Mass in kg.
        """
        return self.dry_mass
    
    # ----------------------------------------------------------------------------------------------------------------------
    def get_wet_mass(self) -> float:
        """Returns the satellite Wet Mass.

        Returns:
            wet_mass (float): satellite Wet Mass in kg.
        """
        return self.wet_mass
    
    # ----------------------------------------------------------------------------------------------------------------------
    def get_mass(self) -> float:
        """Returns the total satellite Mass.

        Returns:
            mass (float): Total satellite Mass in kg.
        """
        return self.dry_mass + self.wet_mass
    
    # ----------------------------------------------------------------------------------------------------------------------
    def get_area(self) -> float:
        """Returns the satellite Area.

        Returns:
            area (float): satellite Area in m^2.
        """
        return self.area
    
    # ----------------------------------------------------------------------------------------------------------------------
    def get_Cd(self) -> float:
        """Returns the satellite Coefficient of Drag.

        Returns:
            Cd (float): satellite Coefficient of Drag.
        """
        return self.Cd
    
    # ----------------------------------------------------------------------------------------------------------------------
    def get_Cr(self) -> float:
        """Returns the satellite Coefficient of Reflectivity.

        Returns:
            Cd (float): satellite Coefficient of Reflectivity.
        """
        return self.Cr
    
    # ----------------------------------------------------------------------------------------------------------------------
    def update_dry_mass(self, dry_mass: float):
        """Redefines the satellite Dry Mass.

        Args:
            dry_mass (float): satellite Dry Mass in kg.
        """
        if not isinstance(dry_mass, int|float) or (dry_mass < 0.0):
            raise TypeError('SatelliteProperties(): dry_mass must be an integer or float greater than 0.0')
        self.dry_mass = dry_mass
    
    # ----------------------------------------------------------------------------------------------------------------------
    def update_wet_mass(self, wet_mass: float):
        """Redefines the satellite Wet Mass.

        Args:
            wet_mass (float): satellite Wet Mass in kg.
        """
        if not isinstance(wet_mass, int|float) or (wet_mass < 0.0):
            raise TypeError('SatelliteProperties(): wet_mass must be an integer or float greater than 0.0')
        self.wet_mass = wet_mass

    # ----------------------------------------------------------------------------------------------------------------------
    def update_area(self, area: float):
        """Redefines the satellite Area.

        Args:
            area (float): satellite Area in m^2.
        """
        if not isinstance(area, int|float) or (area < 0.0):
            raise TypeError('SatelliteProperties(): area must be an integer or float greater than 0.0')
        self.area = area
    
    # ----------------------------------------------------------------------------------------------------------------------
    def update_Cd(self, Cd: float):
        """Redefines the satellite Coefficient of Drag.

        Args:
            Cd (float): satellite Coefficient of Drag.
        """
        if not isinstance(Cd, int|float) or (Cd < 0.0):
            raise TypeError('SatelliteProperties(): Cd must be an integer or float greater than 0.0')
        self.Cd = Cd

    # ----------------------------------------------------------------------------------------------------------------------
    def update_Cr(self, Cr: float):
        """Redefines the satellite Coefficient of Reflectivity.

        Args:
            Cr (float): satellite Coefficient of Reflectivity.
        """
        if not isinstance(Cr, int|float) or (Cr < 0.0):
            raise TypeError('SatelliteProperties(): Cr must be an integer or float greater than 0.0')
        self.Cr = Cr

    # ----------------------------------------------------------------------------------------------------------------------
    def to_dict(self):
        """Method that creates a dictionary of required inputs for SatelliteProperties construction.

        Returns:
            constructor (dict): dictionary of required inputs for SatelliteProperties construction.
        """
        return {
            "type": 'SatelliteProperties',
            "dry_mass": self.dry_mass,
            "wet_mass": self.wet_mass,
            "area": self.area,
            "Cd": self.Cd,
            "Cr": self.Cr,
            }
    
    # ----------------------------------------------------------------------------------------------------------------------
    @staticmethod
    def from_dict(dict):
        """Method that creates a SatelliteProperties from a dictionary of required inputs.

        Args:
            dict (dict): dictionary of required inputs

        Returns:
            sp (SatelliteProperties): SatelliteProperties
        """
        # Check that dictionary of construction is of the correct type
        if dict['type'] != 'SatelliteProperties':
            raise ValueError('SatelliteProperties(): Invalid construction dictionary')

        return SatelliteProperties(dict["dry_mass"], dict["wet_mass"], dict["area"], dict["Cd"], dict["Cr"])
    
    # ----------------------------------------------------------------------------------------------------------------------
    # Operator Methods
    # ----------------------------------------------------------------------------------------------------------------------
    def __eq__(self, other) -> bool:
        """Override Equality operator.
        """
        # Error checking
        if not isinstance(other, SatelliteProperties):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")
        
        # Compare SatelliteProperties
        if isinstance(other, SatelliteProperties):
            return np.all(self._dict__ == other.__dict__)
 
    # ----------------------------------------------------------------------------------------------------------------------
    def __ne__(self, other) -> bool:
        """Override Non-Equality operator.
        """
        # Error checking
        if not isinstance(other, SatelliteProperties):
            raise NotImplementedError(f"Comparison is not defined for {type(other)}")
        
        # Compare SatelliteProperties
        if isinstance(other, SatelliteProperties):
            return np.all(self._dict__ != other.__dict__)
    
    # ----------------------------------------------------------------------------------------------------------------------
    # Representation Methods
    # ----------------------------------------------------------------------------------------------------------------------
    def __str__(self):
        """String Representation of SatelliteProperties Class"""
        class_string = '| '
        class_string += f'Dry Mass={self.dry_mass} | '
        class_string += f'Wet Mass={self.wet_mass} | '
        class_string += f'Area={self.area} | '
        class_string += f'Cd={self.Cr} | '
        class_string += f'Cr={self.Cd} | '

        return class_string
    
    # ----------------------------------------------------------------------------------------------------------------------
    def __repr__(self):
        """Class Representation of SatelliteProperties Class"""
        return f'ForceModel(dry_mass={self.dry_mass}, wet_mass={self.wet_mass}, area={self.area}, Cd={self.Cd}, Cr={self.Cr})'   
        
