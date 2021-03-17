import fill
    from "../../../../../usr/lib/python3.9/site-packages/jupyterlab/staging/yarn";

class Schroedinger{
   // w_x = 2.0 * np.pi * 33.0
    constructor(xSt, xEnd, resN, max_timesteps, dt, w_x, imag_time) {
        let box_x_len = (xEnd - xSt)

        this.xSt = xSt;
        this.xEnd = xEnd;
        this.resN = resN;
        this.max_timesteps = max_timesteps;
        this.dt = dt;
        this.w_x = w_x;
        this.t = 0.0;
        this.dkx = Math.pi / (box_x_len / 2.0);
        this.dx = box_x_len / resN;

        // fft has a different ordering of its x space
        this.kx = np.fft.fftfreq(resN, d=1.0 / ( this.dkx * resN));

        this.k_squared = this.kx ** 2;

        if (imag_time){
            // Convention: $e^{-iH} = e^{UH}$
            this.U = math.complex(-1.0, 0);
        else{
            this.U =  math.complex(0, -1.0);
        }

    }

    get_norm(p){
       return Math.sum(Math.abs(this.psi_val) ** p) * this.dx
    }

    timestep(){
        // create array with x positions
        let zeros = require("zeros");
        this.x = zeros([1, this.resN])
        fill(this.x, function(i, j){
            return this.dx * i + this.xSt;
        })

        // apply functions psi and V to the array of x positions
        this.psi_val = this.psi(this.x);
        this.V_val = this.V(this.x);

        let H_pot = Math.exp(this.U * (0.5 * this.dt) * this.V_val);
        // compute half time step in real space
        this.psi_val = H_pot * this.psi_val;

        // compute full time step in momentum space
        this.H_kin = Math.exp(this.U * (0.5 * this.k_squared) * this.dt);
        this.psi_val = np.fft.fftn(this.psi_val);
        this.psi_val = this.H_kin * this.psi_val;
        this.psi_val = np.fft.ifftn(this.psi_val);

        // update H_pot
        H_pot = Math.exp(this.U * (0.5 * this.dt) * this.V_val);
        // compute half time step in real space
        this.psi_val = H_pot * this.psi_val;

        // normalize
        let psi_norm_after_evolution = this.get_norm(2.0);
        this.psi_val = this.psi_val / Math.sqrt(psi_norm_after_evolution);

        // update time
        this.t = this.t + this.dt
    }
}
