//#include "mps.hpp"
//#include "mpo.hpp"
#include "solver.hpp"
#include "sampler.hpp"
#include <iostream>
#include <complex>
#include <algorithm>
#include <string>
#include <chrono>
#include <iomanip>
#include <Eigen/SVD>
#include <bits/stdc++.h>

using namespace std;

unsigned bond_dim;
unsigned trial_bond_dim; 
unsigned d;
unsigned site_num;      
unsigned num_thread_global=1; 
unsigned krylov_dim=5; 
unsigned num_restart= 10; 

int num_step = 10; 
int num_walker=100; 
int num_rand_h=100; 
int group_size=100; 

double epsilon_lanczos = 1E-6;
double epsilon_sweep = 1E-7; 
double d_tau = 0.01;
double identity_scale = 2.0;

bool is_semi=false;

string filename;
string wf_name; 
string wf_trial_input;
string wf_walker_input; 
string dmrg_trial_wf="random"; 
string job="dmrg";

void parse(string key, string val)
{
    if(key == "filename")
    {
        filename = val;
        cout <<"using input "<< key  << "=" << filename <<endl ;
    }
    else if (key == "dmrg_trial_wf")
    {
        dmrg_trial_wf = val; 
        cout <<"using input "<< key  << "=" << dmrg_trial_wf <<endl ;   
    }
    else if (key == "is_semi")
    {
        if (val == "true")
            is_semi = true; 
        
        cout <<"using input "<< key  << "=" << is_semi <<endl ;
    }
    else if (key == "num_rand_h")
    {
        num_rand_h = stoi(val);
        cout <<"using input "<< key  << "=" << num_rand_h <<endl;
    }
    else if (key== "identity_scale")
    {
        identity_scale = stod(val);
        cout <<"using input "<< key  << "=" << identity_scale <<endl ;
    }
    else if (key=="job")
    {
        job = val;
        cout <<"using input "<< key  << "=" << job <<endl ;
    }
    else if (key=="num_step")
    {
        num_step = stoi(val);
        cout <<"using input "<< key  << "=" << num_step <<endl ;   
    }
    else if (key=="trial_bond_dim")
    {
        trial_bond_dim = stoi(val);
        cout <<"using input "<< key  << "=" << trial_bond_dim <<endl ;
    }
    else if (key=="wf_trial_input")
    {
        wf_trial_input = val; 
        cout <<"using input "<< key  << "=" << wf_trial_input <<endl ;
    }
    else if (key == "wf_walker_input")
    {
        wf_walker_input = val;
        cout <<"using input "<< key  << "=" << wf_walker_input <<endl ;
    }
    else if (key=="num_walker")
    {
        num_walker = stoi(val);
        cout <<"using input "<< key  << "=" << num_walker <<endl ;
    }
    else if (key=="group_size")
    {
        group_size = stoi(val);
        cout <<"using input "<< key  << "=" << group_size <<endl ;
    }
    else if(key == "wf_name")
    {
        wf_name = val;
        cout <<"using input "<< key  << "=" << wf_name <<endl ;
    }
    else if (key == "bond_dim")
    {
        bond_dim = stoi(val);
        cout <<"using input "<< key  << "=" << bond_dim <<endl ;
    }
    else if (key=="site_num")
    {
        site_num = stoi(val);
        cout <<"using input "<< key  << "=" << site_num <<endl ;   
    }
    else if (key=="d")
    {
        d = stoi(val);
        cout <<"using input "<< key  << "=" << d <<endl ;
    }
    else if (key=="num_thread")
    {
        num_thread_global = stoi(val);
        cout <<"using input "<< key  << "=" << num_thread_global <<endl ;
    }
    else if (key=="krylov_dim")
    {
        krylov_dim = stoi(val);
        cout <<"using input "<< key  << "=" << krylov_dim <<endl ;
    }
    else if (key == "num_restart")
    {
        num_restart = stoi(val);
        cout <<"using input "<< key  << "=" << num_restart <<endl ;
    }
    else if (key == "epsilon_lanczos")
    {
        epsilon_lanczos = stod(val);
        cout <<"using input "<< key  << "=" << epsilon_lanczos <<endl ;
    }
    else if (key == "epsilon_sweep")
    {
        epsilon_sweep = stod(val);
        cout <<"using input "<< key  << "=" << epsilon_sweep <<endl ;
    }
    else if (key == "d_tau")
    {
        d_tau = stod(val);
        cout <<"using input "<< key  << "=" << d_tau <<endl ;
    }
    else
    {
        cout << "key "<< key << " not recognized\n";
        //throw invalid_argument("asdsad");
    }

}   

void get_paras(const string& input_name)
{
    /*
        read parameters from sfci.input
    */
    string raw; 

    ifstream file (input_name);
    
    if (file.is_open())
    {

        while(getline(file,raw))
        {
            //position of first appearance
            auto pos = raw.find('=');

            string name (raw.begin(),raw.begin() + pos );
            string val (raw.begin() + pos + 1, raw.end()); 
            parse(name,val);
        }

        file.close(); 
    } 
    
}

void test_parallel()
{
    
    Diag_MPO<double> mpo(site_num,d,num_thread_global,filename,"RHF");
    MPS<double> psi(bond_dim, site_num, d);
    
    psi.rand_init();     
    psi.normalize();
    
    int site_l = 15;
    //int site_l = 3;
    assert(site_l < site_num);
    
    // creating mixed canonical state
    psi.left_canonicalize(0,site_l);
    psi.right_canonicalize(site_l,site_num-1);  // QRing [site_l+1,end]
    //psi.check_mixed_canonical(site_l);  

    vector<MatrixXd> Q;
    vector<MatrixXd> Q_single;

    vector<MatrixXd> M_onsite(d);

    for (int i=0; i< d; i++)
    {
        M_onsite[i] = psi.mat_vec[i][site_l];
    } 

    mpo.apply_mps(psi,M_onsite,Q,site_l);
}

void print_product_state(const MPS<double>& psi)
{
    for(int j=0; j<psi.site_num; j++)
        cout << j << ":" << psi.mat_vec[0][j](0,0) << " " << psi.mat_vec[1][j](0,0) <<endl;
    
    cout << endl;
}

void test_FCI()
{
    /*
        mini-FCI with product states as basis 
    */
    
    Diag_MPO<double> mpo(site_num,d,num_thread_global,filename,"RHF");
    MPS<double> psi(bond_dim, site_num, d);
    MPS<double> phi(bond_dim, site_num, d);

    int size_spinup_sec = site_num/2 * (site_num/2 - 1) /2;
    
    cout << "single section: " << size_spinup_sec<<endl;

    int num_basis = size_spinup_sec*size_spinup_sec;
    
    vector<MPS<double>> basis(num_basis);

    typedef SelfAdjointEigenSolver<MatrixXd> SAES;

    //cout <<mpo.t_prime <<endl;    
    
    SAES saes;
    
    // constructing complete basis

    // 4 electrons 
    int idx =0;
    for (int i_up = 0; i_up < site_num/2; i_up++)
        for (int j_up =i_up+1; j_up< site_num/2; j_up++)
        {
            for (int i_down =0; i_down < site_num/2; i_down++)
                for(int j_down = i_down+1; j_down < site_num/2; j_down++)
                {
                    basis[idx].initialize(bond_dim, site_num, d);
                    for (int k = 0; k <basis[idx].site_num; k++)
                    {
                        basis[idx].mat_vec[0][k] = MatrixXd::Zero(1,1);     // |0>
                        basis[idx].mat_vec[1][k] = MatrixXd::Constant(1,1,1); // |1>

                        if (k == i_up*2 || k == j_up*2 || k == (i_down*2+1) || k == (j_down*2+1) )
                        {
                            basis[idx].mat_vec[0][k](0,0) = 1;  // |0>
                            basis[idx].mat_vec[1][k](0,0) = 0;
                        }
                        
                    }
                    idx++;
                }
        }
        
    cout << "number of basis:" << idx <<endl; 
     
    // 2 electrons 
    /*
    for (int i=0; i<size_spinup_sec; i++)
        for (int j=0; j<size_spinup_sec; j++)
        {
            int idx= i * size_spinup_sec + j;

            int up_pos = 2*i;
            int down_pos = 2*j+1;

            basis[idx].initialize(bond_dim, site_num, d);

            // set 
            for (int k = 0; k <basis[idx].site_num; k++)
            {
                basis[idx].mat_vec[0][k] = MatrixXd::Zero(1,1);     // |0>
                basis[idx].mat_vec[1][k] = MatrixXd::Constant(1,1,1); // |1>

                if (k == up_pos || k == down_pos)
                {
                    basis[idx].mat_vec[0][k](0,0) = 1;  // |0>
                    basis[idx].mat_vec[1][k](0,0) = 0;
                }
                
            }
            
            //print_product_state(basis[idx]); 
        }
    */
    for(int i=0; i< phi.site_num;i++)
    {
        phi.mat_vec[0][i].resize(0,0);
        phi.mat_vec[1][i].resize(0,0);
    }

    MatrixXd H = MatrixXd::Zero(num_basis, num_basis);
    
    for (int i=0; i<num_basis; i++)
        for(int j=i; j<num_basis; j++)
        {
            double sum = 0; 
            cout << "***************************\n";
            //cout << "product state " << j <<endl;;
            print_product_state(basis[j]);
            cout << "calculating(" << i  << "," << j << ")" << endl;

            for (int alpha =0; alpha < mpo.h.size(); alpha++)
            {
                basis[j].apply_single_h(mpo.h[alpha],phi);

                double q  = basis[i].inner_product(phi);
                double temp = mpo.coeff[alpha] * q;

                sum += temp;
                
                /*
                cout <<"with " << mpo.coeff[alpha] << ","<< mpo.label[alpha] << " ," << temp <<endl;
                
                if( abs(q)>1E-12)
                {   
                    
                    cout << "before\n";
                    print_product_state(basis[j]);

                    for (int k=0; k<mpo.h[alpha].size(); k++)
                    {
                        cout << "site " <<k <<endl;
                        cout << mpo.h[alpha][k]<<endl;
                    }
                    cout << "after\n";
                    print_product_state(phi);
                }   
                */
                
            }

            cout << "off-diag element (" <<i << ","<< j << "):" << sum  <<endl;
            H(i,j) = sum;
            H(j,i) = sum; 
            
        }

    //cout << H.row(0) <<endl; 

    saes.compute(H);
    
    auto eigenval = saes.eigenvalues();
    cout << "E_fci =" <<  eigenval(0) <<endl;
}

void test_FCI_diagonal()
{
    Diag_MPO<double> mpo(site_num,d,num_thread_global,filename,"RHF");
    MPS<double> psi(bond_dim, site_num, d);
    MPS<double> phi(bond_dim, site_num, d);

    // check compensation for 1-b ops
    cout <<mpo.t_prime <<endl;

    for(int i=0; i<mpo.h.size(); i++)
        cout << mpo.coeff[i] <<"," << mpo.label[i] <<  endl;
    
    int site_idx = 0;
    // test product states H2
    
    for (int site_idx =1; site_idx <= 4;site_idx++)
    {
        psi.mat_vec.resize(2);
        for (int i=0;i<psi.d;i++)
        {
            psi.mat_vec[i].resize(psi.site_num);
        }
        
        phi.mat_vec.resize(2);
        for (auto& mat:phi.mat_vec)
            mat.resize(phi.site_num);

        // asd
        for(int j=0; j<psi.site_num; j++)
        {    
            psi.mat_vec[0][j] = MatrixXd::Zero(1,1);
            psi.mat_vec[1][j] = MatrixXd::Zero(1,1);

            if ((j == 2*site_idx-2 ) ||  (j == 2*site_idx-1))
            {
                psi.mat_vec[0][j](0,0)=1;
                psi.mat_vec[1][j](0,0)=0;
            }
            else
            {
                psi.mat_vec[0][j](0,0)=0;
                psi.mat_vec[1][j](0,0)=1;
            }
            cout << j << ":" << psi.mat_vec[0][j](0,0) << " " << psi.mat_vec[1][j](0,0) <<endl;
        }    

        double sum = 0; 
        
        for(int alpha = 0 ; alpha < mpo.h.size(); alpha++)
        {
            psi.apply_single_h(mpo.h[alpha],phi);
            
            double q = psi.inner_product(phi);
            double temp =  mpo.coeff[alpha] * q;

            cout <<"with " << mpo.coeff[alpha] << ","<< mpo.label[alpha] << " ," << temp <<endl;
            if(abs(q)>1E-12)
            {
                for (int i=0; i<mpo.h[alpha].size(); i++)
                {
                    cout << "site " <<i <<endl;
                    cout << mpo.h[alpha][i]<<endl;
                }
            }   

            sum +=temp;
        }
        
        cout << site_idx<<endl;
        cout << "product state energy:" <<  sum <<endl;
    } 
        
}

bool my_comp(pair<double,string> a, pair<double,string> b)
{
    return abs(a.first) > abs(b.first); 
}

int main(int argc, char* argv[])
{   
    /*
                
    */
    string input_name; 
    
    if (argv[1]==NULL)
    {
        throw invalid_argument("WARNING: must specify .input file\n");
        //printf("WARNING: must specify .input file\n");
        return 1;
    }
    else
    {
        input_name = argv[1];
        cout << input_name << endl;
    }    
    
    cout <<"***********************************" <<endl; 
    cout <<"***       A NAIVE QC-DMRG       ***" <<endl;
    cout <<"***             and             ***" <<endl;
    cout <<"***  ITS STOCHASTIC ADAPTATION  ***" <<endl;
    cout <<"***          L. Wang            ***" <<endl; 
    cout <<"***      *&%$#%*&*&*@!#?<(      ***" <<endl; 
    cout <<"***********************************" <<endl; 
    cout <<"WARNING: the following line is intentionally left blank\n\n";
    
    cout.precision(15);
    //Eigen::setNbThreads(1);
    get_paras(input_name); 
    
    /*
        You are resposible for ensuring thread_num_global <= OMP_NUM_THREADS
        All Eigen related operations are compiled as SINGLE-THREADED 
    */  
    
    if (job == "dmrg")
    {
        cout << endl; 
        cout <<"***********************************" <<endl; 
        cout <<"***         DMRG starts         ***" <<endl;
        cout <<"***********************************" <<endl; 
        
        DMRG_Solver<double> dmrg_solver(site_num,
                                        d,
                                        bond_dim,
                                        num_thread_global,
                                        num_restart,
                                        krylov_dim,
                                        epsilon_lanczos,
                                        epsilon_sweep,
                                        filename,
                                        dmrg_trial_wf);

        //dmrg_solver.write_wf(); 
        dmrg_solver.sweep();    
    }
    /*
    //dmrg_solver.H.compress_IXYZ(0,0,"9 5 5 0");
    //dmrg_solver.H.transform_to_pauli();

    for (int i=0; i<dmrg_solver.H.h_pauli.size();i++ )
    {
        if (dmrg_solver.H.label_pauli[i] == "const")
        {
            cout << dmrg_solver.H.coeff_pauli[i] <<endl;
        }
    }
    //cout << dmrg_solver.H.coeff_pauli[dmrg_solver.H.h_pauli.size()-1];

    return 0; 
    MPS<double> psi_8, psi_16;
    psi_16.initialize(8,site_num,d);
    //psi_8.initialize(32,site_num,d); 
    //psi_8.read("Li2-6-31g_D=32.wf");

    psi_16.read("Li2-6-31g_D=8.wf");
    
    //cout << "32's norm:" << psi_8.get_norm() <<endl;
    //cout << "16's norm:" << psi_16.get_norm() <<endl;  
    //cout << "overlap:" << psi_16.inner_product(psi_8) << endl;
    
    // energy calculation? 
    
    double e_sum = 0; 
    MPS<double> phi;
    
    phi.initialize(psi_16.bond_dim, psi_16.site_num,psi_16.d); 
    
    for(int i=0; i< dmrg_solver.H.h.size(); i++ )
    {
        psi_16.apply_single_h(dmrg_solver.H.h[i],phi); 
        e_sum += dmrg_solver.H.coeff[i] * psi_16.inner_product(phi);         
    }

    cout << "energy by original:" << e_sum  <<endl; 

    
    e_sum= 0; 

    for(int i=0; i< dmrg_solver.H.h_pauli.size(); i++)
    {
        //cout << dmrg_solver.H.label_pauli[i] <<endl; 
        psi_16.apply_single_h(dmrg_solver.H.h_pauli[i],phi);
        e_sum += dmrg_solver.H.coeff_pauli[i] * psi_16.inner_product(phi);
    }
    
    cout << "energy by pauli: " << e_sum <<endl; 
    
    return 0; 
    */
    else if (job=="mc")
    {
        cout << endl;
        cout <<"***********************************" <<endl; 
        cout <<"***      Monte Carlo starts     ***" <<endl;
        cout <<"***********************************" <<endl; 


        sampler my_sampler( group_size,
                            num_walker,
                            bond_dim,
                            trial_bond_dim, 
                            site_num,
                            d,
                            num_thread_global,
                            num_step,
                            num_rand_h,
                            d_tau,
                            is_semi,
                            filename,
                            wf_trial_input,
                            wf_walker_input
                            );
        // 
        my_sampler.make_group();
        
        
        //my_sampler.run_compressed(); 
        my_sampler.run_test(); 
    }

    else if (job=="dbg")
    {
        /*
            dbg session 
            implement whatever you want 
        */
        std::mt19937_64 random_engine(std::chrono::system_clock::now().time_since_epoch().count());
        /*
        DMRG_Solver<double> dmrg_solver(site_num,
                                        d,
                                        bond_dim,
                                        num_thread_global,
                                        num_restart,
                                        krylov_dim,
                                        epsilon_lanczos,
                                        epsilon_sweep,
                                        filename,
                                        dmrg_trial_wf);
        // identifying 1b and 2b 
        dmrg_solver.H.to_1b_2b();
        */
        
        // Aug 4th begins 

        MPS<double> psi_2(2,site_num,d);
        
        MPS<double> psi_3(4,site_num,d);

        psi_2.read("H2-cc-pvdz_D=2.wf");
        psi_3.read("H2-cc-pvdz_D=4.wf");

        cout << "overlap" << psi_2.inner_product(psi_3) <<endl;
        

        return 0;

        /*
        cout << "before\n";
        cout << phi.mat_vec[0][site_num-1] << endl <<endl;
        cout << phi.mat_vec[0][site_num-2] << endl <<endl;

        phi.right_canonicalize(site_num-2,site_num-1);

        cout << "after\n";
        cout << phi.mat_vec[0][site_num-1] << endl <<endl;
        cout << phi.mat_vec[0][site_num-2] << endl <<endl;
            */

        //return 0; 
        //phi.left_canonicalize(0,24);
    
    }
    else 
    {   
        
    }
    return 0;
}

/*
    @@@ Junkyard @@@
    
    
    auto t1 = std::chrono::system_clock::now(); 
    
    for (int alpha =0; alpha< 20123; alpha+=1007)
    //for (int alpha=0; alpha < mpo.h.size(); alpha++)
    {
        MatrixXd L_tensor, R_tensor; 

        if (site_l > 0) 
            psi.get_L_mat(site_l, mpo.h[alpha], L_tensor);

        if (site_l < site_num-1)
            psi.get_R_mat(site_l, mpo.h[alpha], R_tensor);
        
        // get Q from L and R 
        int num_row = psi.mat_vec[0][site_l].rows();
        int num_col = psi.mat_vec[0][site_l].cols();

        vector<MatrixXd> Q(d);
        Q[0].setZero(num_row,num_col);
        Q[1].setZero(num_row,num_col);
            
        for(int i=0; i<d; i++)
        {
            for (int j=0; j<d; j++ )
            {
                if (site_l ==0 )
                    Q[i] += mpo.h[alpha][site_l](i,j)  * psi.mat_vec[j][site_l] * R_tensor; 
                else if (site_l == site_num -1)
                    Q[i] += mpo.h[alpha][site_l](i,j) * L_tensor * psi.mat_vec[j][site_l]; 
                else
                    Q[i] += mpo.h[alpha][site_l](i,j) * L_tensor * psi.mat_vec[j][site_l] * R_tensor; 
            } 
            //MatrixXd M = mpo.h[alpha][site_l](i,0)*psi.mat_vec[0][site_l] + mpo.h[alpha][site_l](i,1)*psi.mat_vec[1][site_l];
            //Q[i] = L_tensor * M * R_tensor;
        }
        
        double sum = 0;

        for (int i=0; i<d; i++)
        {
            for(int j=0; j< Q[i].cols(); j++)
            {
                sum +=  psi.mat_vec[i][site_l].col(j).adjoint() * Q[i].col(j);
            }   
        }   

        MPS<double> phi(bond_dim, site_num,d);

        psi.apply_single_h(mpo.h[alpha], phi);

        double sum_h = psi.inner_product(phi);

        //cout << "L from get_L\n";
        //cout << L_tensor <<endl;
        cout << "from left and right tensor:" << sum <<endl;
        cout << "from applying h:" << sum_h <<endl<<endl;
        
        
        if (!(abs(sum-sum_h) < 1E-14 ))
        {
            cout << alpha << " fails to match" << endl;
            return 0;
        }
        //cout << (abs(sum-sum_h) < 1E-12 ) <<endl;
        //cout << "from left and right tensor:" << sum <<endl;
        //cout << "from applying h:" << sum_h <<endl<<endl;
        //cout << "the other way" << phi.inner_product(psi)<<endl<<endl;
        
    }

    auto t2 = std::chrono::system_clock::now(); 

    std::chrono::duration<double> t_elapsed = t2-t1;
    std::cout << "execution time:" << t_elapsed.count()<<endl;



    fidelity benchmark 
    
        int bd = 8; 
        int pos = 2;
        double d_t = 0.01;
        
        MPS<double> phi,psi,phi_1,phi_2,phi_1_copy;    
        phi_1.initialize(bond_dim,site_num,d);
        phi_1_copy.initialize(bond_dim,site_num,d);

        phi_1.read(wf_trial_input);
        phi_1_copy.read(wf_trial_input);
        
        //phi_2.read("Li2-6-31g_D=8.wf");
        //phi_1.print();
        //return 0; 

        vector<int> idx_list, idx_sampled; 
        
        std::ranges::sample(idx_list,std::back_inserter(idx_sampled),60,random_engine);

        vector<MPS<double>> results(dmrg_solver.H.idx_1b.size()); 
        
        for (auto& w : results)
        {
            w.initialize(bond_dim,site_num,d);
        }

        for (int i=0; i< dmrg_solver.H.idx_1b.size(); i++)
        {   
            int idx = dmrg_solver.H.idx_1b[i]; 
            
            //cout << idx_sampled[i] <<endl;
            phi_1_copy.apply_single_h(dmrg_solver.H.h[idx],results[i]);
            results[i].multiply_scalar(-d_tau * dmrg_solver.H.coeff[idx]);
            
            MPS<double> total; 
            total.from_direct_sum(phi_1,results[i]);
            total.compress_svd(phi_1_copy.bond_dim);
            
            phi_1.careless_copy(total);
            
        }

        double norm_compressd = phi_1.inner_product(phi_1_copy);
        double norm_original = phi_1_copy.inner_product(phi_1_copy); 
        // now phi_1 is the compressed one  
        for (int i=0; i< results.size(); i++)
        {
            norm_compressd += phi_1.inner_product(results[i]);

            norm_original += (phi_1_copy.inner_product(results[i])+ results[i].inner_product(phi_1_copy) ); 

            for(int j=0; j< results.size(); j++)
            {
                norm_original += results[i].inner_product(results[j]);
            }
        }
        
        // real norm 

        cout << norm_compressd <<endl;
        cout << norm_original <<endl;
        cout << "fidelity: " << norm_compressd / norm_original <<endl; 
        


*/