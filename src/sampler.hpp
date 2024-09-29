#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "mps.hpp"
#include "mpo.hpp"

using namespace std; 

class op_group
{
public: 
    // weight range of each group
    vector<double> range; 

    // idx & actual coefficient 
    vector<vector<pair<int,double>>> idx_op; 

    
};

class sampler
{   
    /*
        sampling the operator based on the two-step importance sampling
    */
public:
    int max_group_size; 
    int num_group;
    int num_walker;
    
    int walker_bd; 
    int num_site;
    int num_rand_h; 
    int num_step;
    int trial_wf_bd; 
    int d; 
    int energy_calc_int; 

    double identity_scale;
    double log10_diff; 
    double d_tau;
    double E_step = 0; 

    unsigned seed; 

    bool is_semi; 

    op_group ite_op; 
    
    uniform_real_distribution<double> real_dice,spawn_dice;
    mt19937_64 gen; 
    
    vector<MPS<double>> walker,trial_buff; 
    vector<double> walker_coeff;

    // trial wave function 
    MPS<double> wf_trial; 

    Diag_MPO<double> mpo_pool; 
    
    sampler(int group_size_val,
            int num_walker_val,
            int walker_bd_val,
            int trial_wf_bd_val, 
            int num_site_val,
            int d_val,
            int num_thread, 
            int num_step_val,
            int num_rand_h_val,         // number of random h_i
            double d_tau_val,   
            bool is_semi_val,           // is 1b applied determinsitically 
            string filename,
            string wf_t_input,
            string wf_walker_input)
    {
        max_group_size = group_size_val;
        num_walker = num_walker_val; 
        num_step = num_step_val; 

        num_rand_h = num_rand_h_val; 
        
        trial_wf_bd = trial_wf_bd_val;
        walker_bd = walker_bd_val;
        num_site = num_site_val;
        d = d_val; 
        
        d_tau = d_tau_val;

        is_semi = is_semi_val; 

        // random_device{}() ?  
        seed = std::chrono::system_clock::now().time_since_epoch().count();
        
        //mt19937 generator(seed);
        gen = mt19937_64(seed);

        // can be reused by calling reset
        real_dice = uniform_real_distribution<double>(0,1); 
        spawn_dice = uniform_real_distribution<double>(0,1);

        // load walkers 
        walker.resize(num_walker);
        for (int i=0; i< num_walker; i++)
        {
            walker[i].initialize(walker_bd,num_site,d);
            if(wf_walker_input=="random")
                walker[i].set_random();
            else
                walker[i].read(wf_walker_input);
        }
        
        
        
        // load trial state 
        wf_trial.initialize(trial_wf_bd,num_site,d);
        wf_trial.read(wf_t_input);

        // init coeff list
        walker_coeff.resize(num_walker);
        for (auto& w: walker_coeff)
            w=1.0;
        
        // load mpo
        // num_of_bucket not used here 
        mpo_pool.initialize(num_site,d,num_thread,filename,"RHF");
        
        //if (is_semi == true) 
        // seperate 2b and 1b 
        mpo_pool.to_1b_2b(); 
        //mpo_pool.transform_to_pauli(); 
        // trial buff
        
        // !do not allocate memory dynamically
        // use mpo_pool.h : less terms 
        trial_buff.resize(mpo_pool.h.size());
        for (auto& w: trial_buff)
            w.initialize(trial_wf_bd, num_site, d); 
    
        // E_step 
        E_step = mpo_pool.calc_expectation(wf_trial);
        cout << "Energy of trial wf: " << E_step <<endl; 
           
    }
    
    ~sampler() {;}
    
    double make_group();    // from mpo.coeff 
    
    void sample_op( vector<pair<int,double>>&,
                    vector<double>&); 
    
    // sum_w <phi_t|w>     
    double get_overlap();

    // sum_w <phi_t| H |w>
    double get_expecation();
    double get_ite_energy();

    void run(); 
    void run_compressed(); 
    void run_test();
};
