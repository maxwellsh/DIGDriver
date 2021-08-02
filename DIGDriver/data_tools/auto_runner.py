import DIG_auto
import sys
import traceback

cancers =['Head-SCC_SNV', 'Adenocarcinoma_tumors_SNV_msi_low','Liver-HCC_SNV', 'Biliary-AdenoCA_SNV',
'Bladder-TCC_SNV', 'Lung-SCC_SNV', 'Bone-Osteosarc_SNV', 'Lung_tumors_SNV','Breast-AdenoCa_SNV', 
'Ovary-AdenoCA_SNV','Carcinoma_tumors_SNV_msi_low','Panc-AdenoCA_SNV',
'CNS-GBM_SNV', 'Pancan_SNV', 'CNS_tumors_SNV','Prost-AdenoCA_SNV',
'ColoRect-AdenoCA_SNV',	'Sarcoma_tumors_SNV','ColoRect-AdenoCA_SNV_msi_low', 'Skin-Melanoma_SNV',
'Digestive_tract_tumors_SNV','Squamous_tumors_SNV','Digestive_tract_tumors_SNV_msi_low', 'Thy-AdenoCA_SNV',
'Eso-AdenoCa_SNV', 'Uterus-AdenoCA_SNV','Female_reproductive_system_tumors_SNV_msi_low','Uterus-AdenoCA_SNV_msi_low']

for c in cancers:
    try:
        fmut_str = '/data/cb/maxas/data/projects/cancer_mutations/cancer_mutations_PCAWG/DIG_FILES/' + c + '.annot.txt.gz'
        gp_str = '/scratch2/dig/full_pcawg/' + c + '/gp_results_fold_{}.h5'
        dig_args = DIG_auto.parse_args('runDIG --out-dir {} --window-size {} --min-map {} --ref-file {} --mut-file {} --N-procs {} --map-file {} --fmodel-dir {} --gp-results-base {} -c {}'.format('/scratch1/priebeo/PCAWG_full_results/v1_final_results', 10000, 0.5, '/scratch1/maxas/ICGC_Roadmap/reference_genome/hg19.fasta', fmut_str, 30, '/scratch1/priebeo/neurIPS/10kb_map_0', '/scratch1/priebeo/PCAWG_full_results/v1_final_results/fmodel_10000_trinuc_192.h5', gp_str, c))
        DIG_auto.run(dig_args)
    except:
        print("Unexpected error:")
        traceback.print_exc()
        print('failed' + c)
        print('skipping...')
