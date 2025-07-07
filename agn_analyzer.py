import numpy as np
import json
import astropy.units as u
from astropy.cosmology import Planck18 as cosmo
from specutils import Spectrum1D, SpectralRegion
from specutils.manipulation import extract_region
from specutils.fitting import fit_lines
from astropy.modeling import models, fitting
import extinction
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.constants import c

from specutils.fitting import fit_continuum


plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'serif'


class AGNAnalyzer:
    def __init__(self, config_path):
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        self.object_name = config["object_name"]
        self.spectrum_path = config["spectrum_path"]
        self.spectrum_abs_path = config["spectrum_abs_path"]
        self.initial_guess = config["initial_guess"]
        self.fit_o_4959 = config.get("fit_o_4959", False)
        self.R = config["R"]
        self.mag_V = config["mag_V"]
        self.mag_err = config["mag_err"]
        self.EBV = config["EBV"]
        self.subtitle_1 = config["subtitle_1"]
        self.subtitle_2 = config["subtitle_2"]
        self.monte_carlo_runs = config.get("monte_carlo_runs", 100)
        self.output_file = config.get("output_file", f"{self.object_name}.png")
        self.z_plot_margin = config.get("z_plot_margin", [200, 400])

        self.spectrum = Spectrum1D.read(self.spectrum_path, format='wcs1d-fits')
        self.spectrum_abs = Spectrum1D.read(self.spectrum_abs_path, format='wcs1d-fits')

    def estimate_redshift(self, spectrum):
        """
        Fit Hβ, [OIII] 5007, and Hα emission lines and estimate redshift and its dispersion.

        Returns
        -------
        z_mean : float
            Mean redshift from all lines.
        z_std : float
            Standard deviation among the individual redshifts.
        """
        lines = []
        lines_approx = []
        stddev = []
        lines_name = []

        for i, line in enumerate(self.initial_guess):
            lines.append(line["rest"])
            lines_name.append(line["name"])
            lines_approx.append(line["mean"])
            stddev.append(line["stddev"])

        z_list = []
        fit_lines = []
        centers = []
        wvl = []


        for i, rest_wave in enumerate(lines):
            width = stddev[i] * 5.0 
            if width < 40:
                width = 40
            region = SpectralRegion(lines_approx[i] * u.AA - width * u.AA,
                                    lines_approx[i] * u.AA + width * u.AA)
            sub_spec = extract_region(spectrum, region)

            flux = sub_spec.flux.value  - 1
            wave = sub_spec.spectral_axis.to_value()
            wvl.append(wave)

            fit_init = models.Gaussian1D(mean=lines_approx[i], stddev=stddev[i])
            fit_init.stddev.bounds = (stddev[i], stddev[i] * 2)
            fit_init.amplitude.bounds = (0.25, None)
            fitter = fitting.LevMarLSQFitter()
            fit = fitter(fit_init, wave, flux)
            fit_lines.append(fit)
            centers.append(fit.mean.value)
            z_i = (fit.mean.value / rest_wave) - 1
            print(f"{lines_name[i]} z = {z_i:.6f}")
            z_list.append(z_i)

        if not z_list:
            raise RuntimeError("No reliable line fits to estimate redshift.")

        return np.mean(z_list), np.std(z_list), z_list, fit_lines, wvl, centers

    def shift_spectrum_to_rest(self, spectrum, z):
        rest_wavelength = spectrum.spectral_axis / (1 + z)
        return Spectrum1D(spectral_axis=rest_wavelength, flux=spectrum.flux)

    def fit_with_uncertainty(self, spectrum, rel_flux_err=None):
        """
        Estimate uncertainties on FWHM and L5100 using Monte Carlo simulations
        with noise drawn from relative error at 5100 Å.

        Parameters
        ----------
        spectrum : Spectrum1D
            Rest-frame extracted region to fit (around Hβ).
        rel_flux_err : float
            Relative error (σ_F / F) derived from V-band magnitude uncertainty.

        Returns
        -------
        fwhm_values : ndarray
            List of FWHM (km/s) from each fit.
        L5100_values : ndarray
            List of fluxes around 5100 Å from each fit.
        """
        n_iter = self.monte_carlo_runs
        flux = spectrum.flux.value
        wave = spectrum.spectral_axis.value

        if rel_flux_err is None:
            rel_flux_err = 0.1  # default: 10%

        fwhm_values = []
        L5100_values = []

        for _ in range(n_iter):
            noise = np.random.normal(0, rel_flux_err * flux)
            noisy_flux = flux + noise

            noisy_spectrum = Spectrum1D(
                spectral_axis=spectrum.spectral_axis,
                flux=noisy_flux * spectrum.flux.unit
            )

            try:
                fit, _ = self.fit(noisy_spectrum)
                fwhm_kms = self.gaussian_fwhm_in_velocity(fit, None)
                if fwhm_kms:
                    fwhm_values.append(fwhm_kms)

                # L5100 measurement (mean flux between 5095-5105 Å)
                region = SpectralRegion(5095 * u.AA, 5105 * u.AA)
                sub = extract_region(noisy_spectrum, region)
                L5100_values.append(np.mean(sub.flux.value))

            except Exception:
                continue

        return np.array(fwhm_values), np.array(L5100_values)

    def fit(self, spectrum):
        flux = spectrum.flux.value
        wave = spectrum.spectral_axis.value

        Hbeta = 4861
        O3_1 = 4959
        O3_2 = 5007

        # Create Gaussian1D models for each of the H-beta and [OIII] lines.
        hbeta_broad = models.Gaussian1D( mean=Hbeta, stddev=10)
        hbeta_broad.mean.bounds = (4800, 4900)
        hbeta_broad.amplitude.bounds = (0.2, None)
        hbeta_narrow = models.Gaussian1D(mean=Hbeta, stddev=5)
        hbeta_narrow.mean.bounds = (4851, 4871) 
        hbeta_narrow.amplitude.bounds = (0, None)
        
        o3_1 = models.Gaussian1D(amplitude=0.25, mean=O3_1, stddev=5)
        o3_2 = models.Gaussian1D(amplitude=0.25, mean=O3_2, stddev=5)

        # Create a polynomial model to fit the continuum.
        mean_flux = flux.mean()
        cont = np.where(flux > mean_flux, mean_flux, flux)
        linfitter = fitting.LinearLSQFitter()
        poly_cont = linfitter(models.Polynomial1D(1), wave, cont)

        # Create a compound model for the four emission lines and the continuum.
        if self.fit_o_4959:
            model = hbeta_narrow + hbeta_broad + o3_1 + o3_2 + poly_cont
        else:
            model = hbeta_narrow + hbeta_broad + o3_2 + poly_cont

        def tie_o3_ampl(model):
            return model.amplitude_3 / 2.98

        o3_1.amplitude.tied = tie_o3_ampl    

        def tie_o3_mean(m):
            return m.mean_0 * O3_2 / Hbeta

        o3_2.mean.tied = tie_o3_mean

        fitter = fitting.TRFLSQFitter()
        fitted_model = fitter(model, wave, flux)

        return fitted_model, fitter

    def plot_fit_result(self, ax, spectrum, model):
        lam = spectrum.spectral_axis.to_value()
        flux = spectrum.flux.value
        fitted_flux = model(lam)

        components = [model[i] for i in range(model.n_submodels)] if hasattr(model, 'n_submodels') else [model]
        ax.plot(lam, flux, color='k', lw=1.1, label='Data')
        ax.plot(lam, fitted_flux, 'r', lw=0.8, label='Total')

        if self.fit_o_4959:
            lines = [r"Narrow H$\beta$", r"Broad H$\beta$", r"[OIII] 4959 $\AA$", r"[OIII] 5007 $\AA$", r"Continuum"]
        else:
            lines = [r"Narrow H$\beta$", r"Broad H$\beta$", r"[OIII] 5007 $\AA$", r"Continuum"]
        for i, comp in enumerate(components):
            ax.plot(lam, comp(lam), "--", lw=1, label=lines[i])
        ax.set_xlabel(r'Wavelength (\AA)')
        ax.set_ylabel(r'Flux')
        ax.set_title(r'Rest-frame multi-Gaussian fit')
        return ax

    def fwhm_instr_from_R(self, lambda_rest):
        c = 299792.458
        delta_lambda = lambda_rest / self.R
        fwhm_instr = c * delta_lambda / lambda_rest
        return fwhm_instr

    def flux_and_error(self, flux_at_5100, mag_err):
        """
        Propagates magnitude uncertainty into flux uncertainty.
        -----
        Zero-point flux (erg/cm^2/s/Å) used to derive the flux from magnitude,
        typically F0 = 3.631e-9 for the AB system in the V band.
        The factor 0.921 comes from the derivative of the flux with respect to magnitude:
            F = F0 * 10^(-0.4 * m)
            dF/dm = -0.4 * ln(10) * F ≈ -0.921 * F
        Therefore:
            flux_err ≈ 0.921 * flux * mag_err
        """
        flux_err = 0.921 * flux_at_5100 * mag_err  # 0.4 * ln(10) ≈ 0.921
        return flux_at_5100, flux_err


    def calculate_mbh(self, f_lambda, fwhm_kms, z, method="vestergaard"):
        """
        Estimate black hole mass using empirical scaling relations.

        Parameters
        ----------
        f_lambda : float
            Observed flux density at 5100 Å in erg/cm²/s/Å.
        fwhm_kms : float
            Full Width at Half Maximum (FWHM) of the broad Hβ line in km/s.
        z : float
            Redshift of the AGN.
        method : str
            Scaling relation to use: either 'vestergaard' or 'feng'.

        Returns
        -------
        log_M : float
            Logarithmic black hole mass in solar masses (log10(M / M_sun)).
        M : float
            Black hole mass in solar masses.
        d_l : float
            Luminosity distance in cm.
        """
        # Compute luminosity distance from redshift
        d_l = cosmo.luminosity_distance(z).to(u.cm).value

        # Correct for Galactic extinction using CCM89
        R_V = 3.1
        wavelength = np.array([5100], dtype=float)
        A_lambda = extinction.ccm89(wavelength, R_V * self.EBV, R_V)[0]
        f_corr = f_lambda * 10**(0.4 * A_lambda)  # Extinction-corrected flux

        # Monochromatic luminosity at 5100 Å: L_lambda = 4π d_L² f_corr
        L_lambda = 4 * np.pi * d_l**2 * f_corr
        log_L = np.log10(5100 * L_lambda / 1e44)  # 5100*L_lambda in erg/s

        # Select coefficients for chosen method
        if method == "vestergaard":
            a, b, c = 6.91, 0.5, 2
            fwhm_term = np.log10(fwhm_kms / 1000)  # convert to units of 10^3 km/s
        elif method == "feng":
            a, b, c = 3.602, 0.504, 1.2
            fwhm_term = np.log10(fwhm_kms)
        else:
            raise ValueError("Method must be either 'vestergaard' or 'feng'.")

        # Apply scaling relation
        log_M = a + b * log_L + c * fwhm_term
        return log_M, 10**log_M, d_l

    def propagate_mass_error(self, fwhm_mean, fwhm_std, L5100_mean, L5100_std, z, sigma_z, method='vestergaard', include_intrinsic=True):
        """
        Propagates uncertainty on log(M_BH) from FWHM and L5100,
        optionally including the intrinsic scatter of the scaling relation.

        Returns
        -------
        sigma_logM : float
            Total uncertainty on log(M_BH) in dex.
        """
        ln10 = np.log(10)
        sigma_logFWHM = fwhm_std / (fwhm_mean * ln10)
        sigma_logL = L5100_std / (L5100_mean * ln10)

        if method == 'vestergaard':
            b, c = 0.5, 2
            intrinsic_scatter = 0.43 if include_intrinsic else 0.0
        elif method == 'feng':
            b, c = 0.504, 1.2
            intrinsic_scatter = 0.35 if include_intrinsic else 0.0
        else:
            raise ValueError("Method must be 'vestergaard' or 'feng'.")
        
        # Additional term from redshift uncertainty via D_L
        if sigma_z is not None and z is not None:
            dL = cosmo.luminosity_distance(z).to(u.cm).value
            delta_dL = (cosmo.luminosity_distance(z + sigma_z).to(u.cm).value -
                        cosmo.luminosity_distance(z - sigma_z).to(u.cm).value) / 2
            sigma_logL_z = (2 * delta_dL / dL) / ln10  # L ~ d_L^2 => log L ~ 2 log d_L
            sigma_logL = np.sqrt(sigma_logL**2 + sigma_logL_z**2)

        sigma_logM = np.sqrt((b * sigma_logL)**2 + (c * sigma_logFWHM)**2 + intrinsic_scatter**2)
        return sigma_logM

    def format_with_error(self, value, error, significant_digits=8):
        exponent = int(np.floor(np.log10(abs(value)))) if value != 0 else 0
        val_scaled = value / 10**exponent
        err_scaled = error / 10**exponent
        return f"({val_scaled:.{significant_digits}g} ± {err_scaled:.{significant_digits}g}) × 10^{exponent}"

    def gaussian_fwhm_in_velocity(self, compound_model, fitter):
        for i, submodel in enumerate(compound_model):
            if isinstance(submodel, models.Gaussian1D) and i == 1:
                fwhm_angstrom = submodel.fwhm
                center = submodel.mean.value
                fwhm_kms = (fwhm_angstrom / center) * c / 1000
                return fwhm_kms
        return None

    def error_on_mass(self, logM, err_logM):
        ln10 = np.log(10)
        M = 10**logM
        err_M = ln10 * M * err_logM
        return err_M
    

    def to_latex(self, string):
        return string.replace("_", "\\")

    def analyze(self):
        mean_z, sigma_z, all_z, fit_lines, wvl, centers = self.estimate_redshift(self.spectrum)
        print(f"z = {mean_z:.6f} ± {sigma_z:.6f}")
        spectrum_rest = self.shift_spectrum_to_rest(self.spectrum, mean_z)

        # Estimate flux at 5100 Å (rest frame)
        flux_region = SpectralRegion(5090 * u.AA, 5110 * u.AA)
        sub_flux = self.shift_spectrum_to_rest(self.spectrum_abs, mean_z)
        sub_flux = extract_region(sub_flux, flux_region)
       
        flux_5100 = np.mean(sub_flux.flux.value)
        flux, flux_err = self.flux_and_error(flux_5100, self.mag_err)   

        # Estimate FWHM and L5100 from Monte Carlo simulation
        region_width = 300 * u.AA
        center = 4900 * u.AA
        flux_region = SpectralRegion((center - region_width),
                                (center + region_width))
        sub_spec = extract_region(spectrum_rest, flux_region)

        fit, fitter = self.fit(sub_spec)
        print("relative flux error:", flux_err / flux_5100)
        fwhm_vals, L5100_vals = self.fit_with_uncertainty(sub_spec, flux_err / flux_5100)

        fwhm_mean = np.mean(fwhm_vals)
        fwhm_std = np.std(fwhm_vals)

        L5100_mean = np.mean(L5100_vals)
        L5100_std = np.std(L5100_vals)

        print(f"FWHM = {fwhm_mean:.1f} ± {fwhm_std:.1f} km/s")
        print(f"L5100 = {self.format_with_error(L5100_mean, L5100_std)} erg/cm²/s/Å")

        fig = plt.figure(figsize=(13, 9))
        gs = GridSpec(2, 2, height_ratios=[1, 1], width_ratios=[1, 1], hspace=0.4, wspace=0.15)

        ax0 = fig.add_subplot(gs[0, :])
        flux_full = self.spectrum_abs.flux.value
        wavelength_full = self.spectrum_abs.spectral_axis.to_value()
        ax0.plot(wavelength_full, flux_full, color='k', lw=1.0)
        ax0.set_xlabel(r'Wavelength (\AA)')
        ax0.set_ylabel(r'Flux in erg/cm2/s/Å')
        ax0.set_title(r'\textbf{'+ self.object_name + r'}' + '\n' + self.subtitle_1 + '\n' + self.subtitle_2 + '\n')

        ax1 = fig.add_subplot(gs[1, 0])
        ax1.plot(self.spectrum.spectral_axis.to_value(), self.spectrum.flux, 'k', lw=0.8, label="Observed")
        ax1.plot(spectrum_rest.spectral_axis.to_value(), spectrum_rest.flux, 'grey', lw=0.8, linestyle='--', label="Rest-frame")
        for i, fit_line in enumerate(fit_lines):
             line_name =  self.initial_guess[i]["name"]
             ax1.plot(wvl[i], fit_line(wvl[i]), label=r' '+self.to_latex(line_name))
        ax1.set_xlabel(r'Wavelength (\AA)')
        ax1.set_ylabel(r'Flux')
        ax1.set_title(
            r'Redshift estimation by fitting emission lines' + '\n' +
            r'$\mathbf{z = %.6f \pm %.6f}$' % (mean_z, sigma_z) ,
            loc='center'
        )

        ax2 = fig.add_subplot(gs[1, 1])
        ax2 = self.plot_fit_result(ax2, sub_spec, fit)
        
        xmin, xmax = self.initial_guess[0]["rest"] - self.z_plot_margin[0], self.initial_guess[-1]["mean"] + self.z_plot_margin[1]
        ax1.set_xlim(xmin, xmax)
        yvals = []

        for line in ax1.lines:
            xdata = line.get_xdata()
            ydata = line.get_ydata()
            mask = (xdata >= xmin) & (xdata <= xmax)
            if np.any(mask):
                yvals.append(ydata[mask])

        # Si on a trouvé des valeurs dans la plage, on les utilise
        if yvals:
            yvals = np.concatenate(yvals)
            ymin, ymax = np.min(yvals), np.max(yvals)
            ax1.set_ylim(ymin, ymax * 1.1)

        # Ensuite, tu peux ajuster manuellement si besoin :
        ymin1, ymax1 = ax1.get_ylim()
        ax1.set_ylim(ymin1, ymax1)

        ymin2, ymax2 = ax2.get_ylim()
        ax2.set_ylim(ymin2, ymax2 * 1.4)

        ax1.legend(ncol=2,fontsize='small', loc='upper left', bbox_to_anchor=(0, 1))
        ax2.legend(ncol=2, fontsize='small', loc='upper left', bbox_to_anchor=(0, 1))

        plt.tight_layout()
        plt.savefig(self.output_file)
        plt.show()

        fwhm_kms = self.gaussian_fwhm_in_velocity(fit, fitter)
        fwhm_inst = self.fwhm_instr_from_R(centers[0])
        fwhm_intrinsic = np.sqrt(fwhm_kms**2 - fwhm_inst**2)

        log_M1, M1, d1 = self.calculate_mbh(flux_5100, fwhm_intrinsic, mean_z, method='vestergaard')
        log_M2, M2, d2 = self.calculate_mbh(flux_5100, fwhm_intrinsic, mean_z, method='feng')

        err_logM_v = self.propagate_mass_error(fwhm_mean, fwhm_std, L5100_mean, L5100_std,
                                            method='vestergaard', sigma_z=sigma_z, z=mean_z)
        err_logM_f = self.propagate_mass_error(fwhm_mean, fwhm_std, L5100_mean, L5100_std,
                                            method='feng', sigma_z=sigma_z, z=mean_z)

        err_M1 = self.error_on_mass(log_M1, err_logM_v)
        err_M2 = self.error_on_mass(log_M2, err_logM_f)
        print(f"Object: {self.object_name}")
        print(f"z = {mean_z:.6f} ± {sigma_z:.6f}")
        print(f"FWHM = {fwhm_kms:.1f} ± {0:.1f} km/s, corrected: {fwhm_intrinsic:.1f}")
        print(f"Flux: {self.format_with_error(flux, flux_err)}")
        print('--- Method 1 : Vestergaard ---')
        print(f"log(M_BH) = {self.format_with_error(log_M1, err_logM_v)}")
        print(f"M_BH      = {self.format_with_error(M1, err_M1)}\n")
        print('--- Method 2 : Feng ---')
        print(f"log(M_BH) = {self.format_with_error(log_M2, err_logM_f)}")
        print(f"M_BH      = {self.format_with_error(M2, err_M2)}\n")

