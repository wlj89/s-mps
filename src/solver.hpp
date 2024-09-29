#include "mps.hpp"
#include "mpo.hpp"
#include <Eigen/Eigen>
#include <complex>

using namespace Eigen;
using namespace std; 


template <typename T> class DMRG_Solver
{

public:
    
    typedef Matrix<T,Dynamic,Dynamic> Mat;
    typedef Matrix<T,Dynamic,1> ColVec;
    typedef Matrix<T,1,Dynamic> RowVec;       
    
    typedef Matrix<int,2,2> PauliMat;       // pauli matrices
    typedef vector<PauliMat> PauliStr;

    typedef vector<Mat> basis_vec;

    MPS<T> psi;
    Diag_MPO<T> H;

    unsigned site_num;
    unsigned d;
    unsigned bond_dim;
    unsigned num_thread;
    unsigned krylov_dim;
    unsigned num_restart;              // 10 is big enough for Lanczos
    double epsilon_lanczos = 1E-7;
    double epsilon_sweep = 1E-9;
    
    string name_suffix; 
    string trial_wf; 

    DMRG_Solver(
                unsigned site_num_val,
                unsigned d_val,
                unsigned bd_val,
                unsigned num_threads_val,
                unsigned num_restart_val,               // max # of restart 
                unsigned krylov_dim_val,                // dimensionality of the triadiagonal matrix 
                double epsilon_lanczos_val,
                double epsilon_sweep_val,
                string filename,
                string dmrg_trial_wf
                                
                )
    {   
        site_num = site_num_val;
        d = d_val;
        bond_dim = bd_val;
        num_thread = num_threads_val;
        num_restart = num_restart_val;
        krylov_dim = krylov_dim_val;
        
        epsilon_lanczos = epsilon_lanczos_val;
        epsilon_sweep = epsilon_sweep_val;

        // how do you take initial guess? 
        psi.initialize(bond_dim,site_num,d);
        
        H.initialize(site_num,d,num_thread,filename,"RHF");
        
        name_suffix = filename; 

        trial_wf = dmrg_trial_wf; 
    }
    
    void to_M_onsite(   basis_vec& M,
                        const MPS<T>& psi, 
                        const int site_idx)
    {
        
        M.resize(psi.d);

        // Matrix copy operator allocates space for you 
        for (int i=0; i< psi.d; i++)
        {   
            M[i] = psi.mat_vec[i][site_idx];
        }       
    }
    
    T overlap(const basis_vec& a, const basis_vec& b)
    {
        T sum = 0; 
        for (int i=0; i<d; i++)
        {
            for(int j=0; j<b[i].cols(); j++)
                sum += a[i].col(j).adjoint() * b[i].col(j);
        }
        return sum;
    }
    
    void add_one(   basis_vec& result, 
                    const basis_vec& a,
                    T coeff_b,
                    const basis_vec& b)
    {
        /*
            result = a + coeff_b*b
        */

        for(int i=0; i<d; i++)
        {
            result[i] = a[i] + coeff_b * b[i];  
        }
    }
    
    void add_two(   basis_vec& result, 
                    const basis_vec& a,
                    T coeff_b,
                    const basis_vec& b,
                    T coeff_c,
                    const basis_vec& c)
    {
        /*
            result = a + coeff_b*b + coeff_c*c
        */
        for(int i=0; i<d; i++)
        {
            result[i] = a[i] + coeff_b * b[i] + coeff_c * c[i];  
        }
    }

    void add_many(  basis_vec& M, 
                    const vector<basis_vec>& basis,
                    const ColVec& coeff)
    {
        for (int i=0; i <d; i++)
        {
            M[i].setZero();

            for(int j=0; j < basis.size(); j++)
            {
                M[i] += coeff(j) * basis[j][i];
            }
        }
    }
    
    void normalize(basis_vec& M)
    {
        T norm = overlap(M,M);
        norm = sqrt(norm);
        
        for (auto& item: M)
            item /= norm;
    }
    
    T solve(unsigned site_l)
    {   
        /* 
            solve for paras on site l that minimizes E 
            A Lanczos process under disguise, more efficient than 
            explicitly constructing sparse matrix and solve 

            actual krylov_dim used should not exceed the total 
            number of parameter on site 
        
        */
        typedef SelfAdjointEigenSolver<MatrixXd> SAES;
        SAES saes;

        unsigned num_site_para = psi.mat_vec[0][site_l].cols() *  psi.mat_vec[0][site_l].rows() * psi.d; 
        unsigned krylov_dim_actual = num_site_para < krylov_dim ? num_site_para : krylov_dim; 

        cout << "using Krylov dim " << krylov_dim_actual <<endl;

        // tridiagonal lanczos matrix
        Mat H_eff = Mat::Zero(krylov_dim_actual,krylov_dim_actual);

        // only need the as on site matrix as input
        // Lanczos basis
        vector<basis_vec> basis(krylov_dim_actual); 

        for (int i=0; i< krylov_dim_actual; i++ )
        {
            basis[i].resize(d);
        }
        
        basis_vec M_onsite;
        basis_vec Q;  
        
        // |v0> 
        // operator = just do the job
        to_M_onsite(M_onsite,psi,site_l);
        normalize(M_onsite);

        T energy_prev = 1E12;
        T energy_final; 

        for (int res = 0; res < num_restart; res++)
        {
            // randomizing inital guess ? 
            // DO NOT FOREGET normalize |v0>
            basis[0] = M_onsite;
            
            //cout << "check |v0>\n";
            //cout<< M_onsite[0] <<endl<<endl;
            //cout<< M_onsite[1] <<endl<<endl;
            //normalize(basis[0]);
            
            //basis[0][0] - Mat::Random(M);            
            // H|v0>
            H.apply_mps(psi,basis[0],Q,site_l);
            
            //a0
            H_eff(0,0) = overlap(Q,basis[0]); 
            cout << "KRYLOV SUBSPACE EXPANSION " << res <<endl;
            cout << "ENERGY OF TRIAL WF:"<< H_eff(0,0) <<endl;

            // strange case: energy does not decrease 
            if (abs(H_eff(0,0) - energy_prev) < epsilon_lanczos || H_eff(0,0) > energy_prev)
            {
                // time to stop
                cout << "ENERGY DECREASE BELOW EPSILON. LANCZOS PROCESS HALT.\n\n";

                for (int i=0; i<d; i++)
                {
                    psi.mat_vec[i][site_l] = M_onsite[i];
                }

                return H_eff(0,0);
            }
            else 
            {
                energy_prev = H_eff(0,0);
            }
            
            // get nromalized |v1>
            add_one(basis[1],Q,-H_eff(0,0),basis[0]);

            normalize(basis[1]); 
            
            // b1
            H_eff(0,1) = overlap(Q,basis[1]); // <v0|H|v1>
            //H_eff(1,0) = conj(H_eff(0,1));
            H_eff(1,0) = H_eff(0,1);
            
            //cout << "beep1\n";
            //Q = H |v1> 
            H.apply_mps(psi,basis[1],Q,site_l);
            
            // a1
            H_eff(1,1) = overlap(basis[1],Q);
            
            //cout << "beep2\n";
            // Lanczos tri-diagonalization
            for (unsigned i = 2; i < krylov_dim_actual; i++ )
            {   
                /*  
                    reuse L and R for energy estimation? 
                */
                add_two(basis[i],Q,-H_eff(i-1,i-1),basis[i-1],-H_eff(i-2,i-1),basis[i-2]);
                normalize(basis[i]);    
                
                // b_i
                H_eff(i-1,i) = overlap(Q,basis[i]);
                //H_eff(i,i-1) = conj(H_eff(i-1,i));
                H_eff(i,i-1) = H_eff(i-1,i);        /// only real for now 
                
                // a_i
                H.apply_mps(psi,basis[i],Q,site_l);
                H_eff(i,i) = overlap(basis[i],Q);
                
            }
            // start solving

            saes.compute(H_eff);

            //cout << H_eff <<endl; 

            auto eigenvec = saes.eigenvectors();
            auto eigenval = saes.eigenvalues();
            
            //cout << "Energy from eigensolver: " <<endl<< eigenval <<endl<<endl;;
            energy_final = eigenval(0);

            add_many(M_onsite,basis,eigenvec.col(0));
            normalize(M_onsite);
                        
        }
        // now, restart      
        
        // modify M_onsite and return Epsilon!
        cout << "EPSILON NOT REACHED BEFORE MAX RESTARTS. LANCZOS PROCESS HALT.\n\n";
        
        for (int i=0; i<d; i++)
        {
            psi.mat_vec[i][site_l] = M_onsite[i];
        }
        
        return energy_final;      

    }   
    void write_wf()
    {
        /*
            write 
        */
        string name (name_suffix.begin(), name_suffix.begin() + (name_suffix.length() -11 ));
        name = name  + "_D=" + to_string(bond_dim)+".wf";
        //cout << name <<endl; 
        cout << "WRITING .WF INTO " + name; 
        psi.write(name); 
    } 
    void sweep()
    {   
        /*
            Sweeping routine. 
            Watch out the corner cases : l = 0 & l = site_num-1
        */
        cout << "DMRG SWEEPING STARTS\n\n";
        
        // start from the middle might be better? 
        int site_idx = site_num/2;
        int shift = -1;

        T energy = 1E+12; 

        // start from site 0
        if (trial_wf=="random") 
        {
            cout << "USING RANDOM TRIAL WF...\n";
            psi.set_random();
            //psi.normalize();
        }
        else
        {   
            cout << "USING TRIAL WF FROM " << trial_wf <<endl; 

            psi.initialize(bond_dim,site_num,d);
            psi.set_zero(); 

            psi.read(trial_wf);  


            cout <<"ENERGY INPUT OF TRIAL WF:" << H.calc_expectation(psi) <<endl; 
        }
        //psi.set_uniform();

        psi.normalize(); 
        
        // mixed canonical w.r.t. 
        psi.left_canonicalize(0,site_idx);
        psi.right_canonicalize(site_idx,site_num-1);

        //cout <<"NORM OF INIT WF:" << psi.get_norm() << endl;
        //cout << "canonicalization check:" << psi.check_mixed_canonical(site_idx)<<endl;
        
        while(1)    
        {
            cout<< "*****************************\n";
            cout<< "*** OPTIMIZING SITE " << site_idx <<endl;
            cout<< "*****************************\n";
            
            T energy_now = solve(site_idx);

            cout<< "*****************************\n";
            cout<< "*** ENERGY AFTER OPTIMIZING SITE " << site_idx << ":"; 
            cout<< energy_now << endl;
            cout<< "*****************************\n";

            // sites at two ends have very few degree of freedom 
            // can raise false alarm 
            // one quarter of site_num as a protocol?   
            if ( (abs(energy_now - energy) < epsilon_sweep) && (site_idx >= (3*site_num/8) ) && (site_idx < (5*site_num/8)) ) 
            {
                cout << "DMRG HALTS WITH E=" << energy_now <<endl;
                write_wf(); 
            
                return; 
            }
            else 
            {
                energy = energy_now;
            }
            
            if ((site_idx == site_num-1) || (site_idx==0))
            {
                shift *= -1;
            }

            if (shift>0)
            {
                psi.left_canonicalize(site_idx,site_idx+shift);
            }
            else 
            {
                psi.right_canonicalize(site_idx+shift,site_idx);
            }
            // left or right canonicalize 
            site_idx += shift;

            /*
            cout << "canonicalization check:" << psi.check_mixed_canonical(site_idx)<<endl;
            cout << "norm:" << psi.get_norm() <<endl; 
            */
            //if (site_idx == site_num-1)
            //    return;
        }   
        
    }
};