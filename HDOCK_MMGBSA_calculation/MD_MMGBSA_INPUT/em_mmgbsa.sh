#!/bin/bash

pathalfa=$PWD
NT=$1
GPU=$2

if [[ -z "${NT}" || -z "${GPU}" ]]; then 
echo "
missing INPUT and/or NT and/or GPU variable;
usage: ./em_mmgbsa.sh input.gro nt_number gpu_id_number
"
exit 0
fi

for X in model_*.pdb; do
    
    mkdir $(echo $X | cut -d "." -f1)
    cp $X $(echo $X | cut -d "." -f1)
    cd $pathalfa/$(echo $X | cut -d "." -f1)
    
    pathbeta=$PWD
    mkdir topology
    cd $pathbeta/topology
    echo -e "6 \n 1" | gmx pdb2gmx -f ../model_*.pdb -ignh
    cd $pathbeta/

    gmx editconf -f topology/conf.gro -o box.gro -c -bt dodecahedron -d 0.9
    gmx solvate -cp box.gro -cs -o solv.gro -p topology/topol.top

    gmx grompp -f ../mdp_AA/ions.mdp -c solv.gro -p topology/topol.top -o ions.tpr -maxwarn 1
    echo SOL | gmx genion -s ions.tpr -o ions.gro -p topology/topol.top -pname NA -nname CL -neutral -conc 0.15

    echo "
    keep 1
    splitch 0
    q" | gmx make_ndx -f ions.gro

    # energy minimization 1 #

    mkdir em1
    gmx grompp -f ../mdp_AA/1em.mdp -c ions.gro -p topology/topol.top -o em1/topol.tpr -po em1/mdout.mdp

    cd $pathbeta/em1
    gmx mdrun -v -gpu_id $GPU -nt $NT -s topol.tpr
    cd $pathbeta

    echo potential | gmx energy -f em1/ener.edr -o em1/potential.xvg

    # energy minimization 2 #

    mkdir em2
    gmx grompp -f ../mdp_AA/2em.mdp -c em1/confout.gro -p topology/topol.top -o em2/topol.tpr -po em2/mdout.mdp

    cd $pathbeta/em2
    gmx mdrun -v -gpu_id $GPU -nt $NT -s topol.tpr
    echo Protein | gmx trjconv -s topol.tpr -f confout.gro -n ../index.ndx -o mol.xtc -pbc mol -ur compact
    cd $pathbeta

    echo potential | gmx energy -f em2/ener.edr -o em2/potential.xvg

    # MMGBSA calculation #

    mkdir MMGBSA
    cd $pathbeta/MMGBSA
    cp -r $pathalfa/../../MD_MMGBSA_INPUT/input.in .
    gmx_MMPBSA -i input.in -cs ../em2/topol.tpr -ci ../index.ndx -cg 1 2 -ct ../em2/mol.xtc -nogui
    cd $pathalfa

done

cat model_*/MMGBSA/FINAL_RESULTS_MMPBSA.dat | grep ΔTOTAL | awk '{print $2}' | sort -h > MMGBSA_10_sorted.dat

exit 0





