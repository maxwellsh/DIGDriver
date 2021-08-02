#!/usr/bin/env Rscript

#' mutationFunction
#'
#' Annotation of the genic function of mutations
#'
#' @author Maxwell Sherman; directly adapted from Inigo Martincorena's dNdScv package
#' @param mutations Table of mutations (5 columns: sampleID, chr, pos, ref, alt). Only list independent events as mutations.
#' @param refdb Reference database (path to .rda file)
#' @param max_muts_per_gene_per_sample If n<Inf, arbitrarily the first n mutations by chr position will be kept
#' @param max_coding_muts_per_sample Hypermutator samples often reduce power to detect selection
#'
#' @return 'annot' a dataframe with columns: CHROM, START, END, REF, ALT, SAMPLE, GENE, ANNOT
#' @export

suppressPackageStartupMessages(library(GenomicRanges, quietly=T, warn.conflicts=F))
suppressPackageStartupMessages(library(Biostrings, quietly=T, warn.conflicts=F))
suppressPackageStartupMessages(library(BiocGenerics, quietly=T, warn.conflicts=F))
suppressPackageStartupMessages(library(parallel, quietly=T, warn.conflicts=F))
suppressPackageStartupMessages(library(seqinr, quietly=T, warn.conflicts=F))

mutationFunction = function(mutations, refdb, max_muts_per_gene_per_sample = 3, max_coding_muts_per_sample = 3000) {

    ## 1. Environment
    message(sprintf("Annotating the genic function of %i mutations...", nrow(mutations)))

    mutations = mutations[,1:5] # Restricting input matrix to first 5 columns
    mutations[,c(1,2,3,4,5)] = lapply(mutations[,c(1,2,3,4,5)], as.character) # Factors to character
    mutations[[3]] = as.numeric(mutations[[3]]) # Chromosome position as numeric
    mutations = mutations[mutations[,4]!=mutations[,5],] # Removing mutations with identical reference and mutant base
    mutations = unique(mutations) # Drop duplicate entries (same mutation & sample)
    colnames(mutations) = c("sampleID","chr","pos","ref","mut")
    
    # Removing NA entries from the input mutation table
    indna = which(is.na(mutations),arr.ind=T)
    if (nrow(indna)>0) {
        mutations = mutations[-unique(indna[,1]),] # Removing entries with an NA in any row
        warning(sprintf("%0.0f rows in the input table contained NA entries and have been removed. Please investigate.",length(unique(indna[,1]))))
    }
    
    # [Input] Reference database
    load(refdb)
    gene_list = sapply(RefCDS, function(x) x$gene_name) # All genes [default]

    # Expanding the reference sequences [for faster access]
    for (j in 1:length(RefCDS)) {
        RefCDS[[j]]$seq_cds = base::strsplit(as.character(RefCDS[[j]]$seq_cds), split="")[[1]]
        RefCDS[[j]]$seq_cds1up = base::strsplit(as.character(RefCDS[[j]]$seq_cds1up), split="")[[1]]
        RefCDS[[j]]$seq_cds1down = base::strsplit(as.character(RefCDS[[j]]$seq_cds1down), split="")[[1]]
        if (!is.null(RefCDS[[j]]$seq_splice)) {
            RefCDS[[j]]$seq_splice = base::strsplit(as.character(RefCDS[[j]]$seq_splice), split="")[[1]]
            RefCDS[[j]]$seq_splice1up = base::strsplit(as.character(RefCDS[[j]]$seq_splice1up), split="")[[1]]
            RefCDS[[j]]$seq_splice1down = base::strsplit(as.character(RefCDS[[j]]$seq_splice1down), split="")[[1]]
        }
    }
    
    
    ## 2. Mutation annotation
    nt = c("A","C","G","T")
    
    # Start and end position of each mutation
    mutations$end = mutations$start = mutations$pos
    l = nchar(mutations$ref)-1 # Deletions of multiple bases
    mutations$end = mutations$end + l
    ind = substr(mutations$ref,1,1)==substr(mutations$mut,1,1) & nchar(mutations$ref)>nchar(mutations$mut) # Position correction for deletions annotated in the previous base (e.g. CA>C)
    mutations$start = mutations$start + ind
    
    # Mapping mutations to genes
    ind = setNames(1:length(RefCDS), sapply(RefCDS,function(x) x$gene_name))
    gr_genes_ind = ind[gr_genes$names]
    gr_muts = GenomicRanges::GRanges(mutations$chr, IRanges::IRanges(mutations$start,mutations$end))
    ol = as.data.frame(GenomicRanges::findOverlaps(gr_muts, gr_genes, type="any", select="all"))

    # Extract non-coding mutations
    mutations_nc = mutations[-ol[,1],]
    snv = (mutations_nc$ref %in% nt & mutations_nc$mut %in% nt)
    indels_nc = mutations_nc[!snv,]
    mutations_nc = mutations_nc[snv,]
    # print(nrow(mutations_nc))
    # print('Extracted non-coding mutations')

    # Extract coding mutations
    mutations = mutations[ol[,1],] # Duplicating subs if they hit more than one gene
    mutations$geneind = gr_genes_ind[ol[,2]]
    mutations$gene = sapply(RefCDS,function(x) x$gene_name)[mutations$geneind]
    mutations = unique(mutations)
    # print(nrow(mutations))

    # print('Separated coding and non-coding mutations')

    
    # # Optional: Excluding samples exceeding the limit of mutations/sample [see Default parameters]
    # nsampl = sort(table(mutations$sampleID))
    # exclsamples = NULL
    # if (any(nsampl>max_coding_muts_per_sample)) {
    #     message(sprintf('    Note: %0.0f samples excluded for exceeding the limit of mutations per sample (see the max_coding_muts_per_sample argument in dndscv). %0.0f samples left after filtering.',sum(nsampl>max_coding_muts_per_sample),sum(nsampl<=max_coding_muts_per_sample)))
    #     exclsamples = names(nsampl[nsampl>max_coding_muts_per_sample])
    #     mutations = mutations[!(mutations$sampleID %in% names(nsampl[nsampl>max_coding_muts_per_sample])),]
    # }
    # 
    # # Optional: Limiting the number of mutations per gene per sample (to minimise the impact of unannotated kataegis and other mutation clusters) [see Default parameters]
    # mutrank = ave(mutations$pos, paste(mutations$sampleID,mutations$gene), FUN = function(x) rank(x))
    # exclmuts = NULL
    # if (any(mutrank>max_muts_per_gene_per_sample)) {
    #     message(sprintf('    Note: %0.0f mutations removed for exceeding the limit of mutations per gene per sample (see the max_muts_per_gene_per_sample argument in dndscv)',sum(mutrank>max_muts_per_gene_per_sample)))
    #     exclmuts = mutations[mutrank>max_muts_per_gene_per_sample,]
    #     mutations = mutations[mutrank<=max_muts_per_gene_per_sample,]
    # }
    
    # Additional annotation of substitutions
    
    mutations$strand = sapply(RefCDS,function(x) x$strand)[mutations$geneind]
    snv = (mutations$ref %in% nt & mutations$mut %in% nt)
    # if (!any(snv)) { stop("Zero coding substitutions found in this dataset. Unable to run dndscv. Common causes for this error are inputting only indels or using chromosome names different to those in the reference database (e.g. chr1 vs 1)") }
    indels = mutations[!snv,]
    mutations = mutations[snv,]
    mutations$ref_cod = mutations$ref
    mutations$mut_cod = mutations$mut
    compnt = setNames(rev(nt), nt)
    isminus = (mutations$strand==-1)
    mutations$ref_cod[isminus] = compnt[mutations$ref[isminus]]
    mutations$mut_cod[isminus] = compnt[mutations$mut[isminus]]

    message(sprintf("\t%i\tcoding SNVs", nrow(mutations)))
    message(sprintf("\t%i\tnoncoding SNVs", nrow(mutations_nc)))
    message(sprintf("\t%i\tcoding INDELs", nrow(indels)))
    message(sprintf("\t%i\tnoncoding INDELs", nrow(indels_nc)))

    
    # Subfunction: obtaining the codon positions of a coding mutation given the exon intervals
    
    chr2cds = function(pos,cds_int,strand) {
        if (strand==1) {
            return(which(unlist(apply(cds_int, 1, function(x) x[1]:x[2])) %in% pos))
        } else if (strand==-1) {
            return(which(rev(unlist(apply(cds_int, 1, function(x) x[1]:x[2]))) %in% pos))
        }
    }
    
    # Annotating the functional impact of each substitution and populating the N matrices
    
    if ( nrow(mutations) > 0 ) {
        ref3_cod = mut3_cod = wrong_ref = aachange = ntchange = impact = codonsub = array(NA, nrow(mutations))
        
        for (j in 1:nrow(mutations)) {
        
            geneind = mutations$geneind[j]
            pos = mutations$pos[j]
            
            if (any(pos == RefCDS[[geneind]]$intervals_splice)) { # Essential splice-site substitution
            
                impact[j] = "Essential_Splice"; impind = 4
                pos_ind = (pos==RefCDS[[geneind]]$intervals_splice)
                cdsnt = RefCDS[[geneind]]$seq_splice[pos_ind]

            } else { # Coding substitution
            
                pos_ind = chr2cds(pos, RefCDS[[geneind]]$intervals_cds, RefCDS[[geneind]]$strand)
                cdsnt = RefCDS[[geneind]]$seq_cds[pos_ind]
                codon_pos = c(ceiling(pos_ind/3)*3-2, ceiling(pos_ind/3)*3-1, ceiling(pos_ind/3)*3)
                old_codon = as.character(as.vector(RefCDS[[geneind]]$seq_cds[codon_pos]))
                pos_in_codon = pos_ind-(ceiling(pos_ind/3)-1)*3
                new_codon = old_codon; new_codon[pos_in_codon] = mutations$mut_cod[j]
                old_aa = seqinr::translate(old_codon, numcode = 1)
                new_aa = seqinr::translate(new_codon, numcode = 1)
            
                # Annotating the impact of the mutation
                if (new_aa == old_aa){ 
                    impact[j] = "Synonymous"; impind = 1
                } else if (new_aa == "*"){
                    impact[j] = "Nonsense"; impind = 3
                } else if (old_aa != "*"){
                    impact[j] = "Missense"; impind = 2
                } else if (old_aa=="*") {
                    impact[j] = "Stop_loss"; impind = NA
                }
            }
            
            if (mutations$ref_cod[j] != as.character(cdsnt)) { # Incorrect base annotation in the input mutation file (the mutation will be excluded with a warning)
                wrong_ref[j] = 1
            }
          
            if (round(j/1e4)==(j/1e4)) { message(sprintf('    %0.3g%% ...', round(j/nrow(mutations),2)*100)) }
        }
        
        mutations$impact = impact
        
        if (any(!is.na(wrong_ref))) {
            if (mean(!is.na(wrong_ref)) < 0.1) { # If fewer than 10% of mutations have a wrong reference base, we warn the user
                warning(sprintf('%0.0f (%0.2g%%) mutations have a wrong reference base.', sum(!is.na(wrong_ref)), 100*mean(!is.na(wrong_ref))))
            } else { # If more than 10% of mutations have a wrong reference base, we stop the execution (likely wrong assembly or a serious problem with the data)
                stop(sprintf('%0.0f (%0.2g%%) mutations have a wrong reference base. Please confirm that you are not running data from a different assembly or species.', sum(!is.na(wrong_ref)), 100*mean(!is.na(wrong_ref))))
            }
            wrong_refbase = mutations[!is.na(wrong_ref), 1:5]
            mutations = mutations[is.na(wrong_ref),]
        }
    }

    ## Handle non-coding SNVs
    if (nrow(mutations_nc) > 0) {
        mutations_nc['gene'] = '.'
        mutations_nc['impact'] = 'Noncoding'

        cols = c('sampleID', 'chr' , 'start', 'end', 'ref', 'mut', 'gene', 'impact')
        annot = rbind(mutations[, cols], mutations_nc[, cols])
    } else if ( nrow(mutations) > 0 ) {
        cols = c('sampleID', 'chr' , 'start', 'end', 'ref', 'mut', 'gene', 'impact')
        annot = mutations[, cols]
    } else {
        cols = c('sampleID', 'chr' , 'start', 'end', 'ref', 'mut', 'gene', 'impact')
        annot =  data.frame(sampleID=character(), chr=character(), start=double(), end=double(), 
                    ref=character(), mut=character(), gene=character(), impact=character()
        )
    }
    
    ## Handle coding indels
    # if (any(nrow(indels)) | any(nrow(indels_nc))) { # If there are indels we concatenate the tables of subs and indels
    if (any(nrow(indels))) { # If there are indels we concatenate the tables of subs and indels
        ## EXCLUDE INDELS FOR NOW!
        # warning(sprintf("DIGDriver does not currently support indels. Skipping %i genic and %i noncoding indels", nrow(indels), nrow(indels_nc)))
        # indels = cbind(indels, data.frame(impact="no-SNV"))
        
        # Annotation of indels
        ins = nchar(gsub("-","",indels$ref))<nchar(gsub("-","",indels$mut))
        del = nchar(gsub("-","",indels$ref))>nchar(gsub("-","",indels$mut))
        multisub = nchar(gsub("-","",indels$ref))==nchar(gsub("-","",indels$mut)) # Including dinucleotides
        l = nchar(gsub("-","",indels$ref))-nchar(gsub("-","",indels$mut))
        indelstr = rep('cds_INDEL',nrow(indels))
        for (j in 1:nrow(indels)) {
            geneind = indels$geneind[j]
            pos = indels$start[j]:indels$end[j]
            if (ins[j]) { pos = c(pos-1,pos) } # Adding the upstream base for insertions
            pos_ind = chr2cds(pos, RefCDS[[geneind]]$intervals_cds, RefCDS[[geneind]]$strand)
            if (length(pos_ind)>0) {
                inframe = (length(pos_ind) %% 3) == 0
                if (ins[j]) { # Insertion
                    indelstr[j] = sprintf("INDEL_%0.0f_%0.0f_ins%s",min(pos_ind),max(pos_ind),c("frshift","inframe")[inframe+1])
                } else if (del[j]) { # Deletion
                    indelstr[j] = sprintf("INDEL_%0.0f_%0.0f_del%s",min(pos_ind),max(pos_ind),c("frshift","inframe")[inframe+1])
                } else { # Dinucleotide and multinucleotide changes (MNVs)
                    indelstr[j] = sprintf("INDEL_%0.0f_%0.0f_mnv",min(pos_ind),max(pos_ind))
                }
            }
        }
        indels$impact = indelstr
        annot = rbind(annot, indels[, cols])
    } # else {
      #   annot = mutations
    # }

    # Handle noncoding indels
    if (nrow(indels_nc) > 0) {
        indels_nc['gene'] = '.'
        indels_nc['impact'] = 'Noncoding_INDEL'

        # cols = c('sampleID', 'chr' , 'start', 'end', 'ref', 'mut', 'gene', 'impact')
        annot = rbind(annot, indels_nc[, cols])
    }


        
    annot$start = annot$start - 1  ## Zero-index the mutations
    annot = annot[order(annot$chr, annot$start, annot$end),]

    cols = c('SAMPLE', 'CHROM' , 'START', 'END', 'REF', 'ALT', 'GENE', 'ANNOT')
    cols_reorder = c('CHROM' , 'START', 'END', 'REF', 'ALT', 'SAMPLE', 'GENE', 'ANNOT')
    colnames(annot) = cols
    annot = annot[, cols_reorder]

    return(annot)

}

main = function() {
    args = commandArgs(trailingOnly=TRUE)
    if (length(args) != 3) {
        stop("USAGE: ./mutationFunction <MUTATION FILE> <RDA GENOME FILE> <OUTPUT FILE>")
    }

    fmut = args[1]
    refdb = args[2]
    out = args[3]

    message("Loading mutation data...")
    df_mut = read.table(fmut)
    if (ncol(df_mut) == 5) {  ## ASSUME format is CHROM, POS, REF, ALT, SAMPLE
        df_mut = df_mut[ , c(5, 1, 2, 3, 4)]
    } else {  ## ASSUME format is CHROM, START, END, REF, ALT, SAMPLE
        # df_mut = df_mut[ , c(6, 1, 3, 4, 5)]
        df_mut = df_mut[ , c(6, 1, 2, 4, 5)]
        df_mut[ , 3] = df_mut[, 3] + 1  ## 1-index start position
    }
    colnames(df_mut) = c('sampleID', 'chr', 'pos', 'ref', 'mut')

    annot = mutationFunction(df_mut, refdb)

    if (nrow(annot) > nrow(df_mut)) {
        message(sprintf('WARNING: %i mutation entries were duplicated because they overlap multiple genes. DIG tools will account for this automatically, but beware if using this file with other software.', nrow(annot) - nrow(df_mut)))
    }

    # if (! substr(out, nchar(out)-2, nchar(out)) == '.gz') {
    #     out = past0(out, '.gz')
    # }
    # gz = gzfile(out, 'w')
    write.table(annot, out, col.names=F, row.names=F, quote=F, sep="\t")
}

if (sys.nframe() == 0){
    main()
}

# if (!interactive()) {
#   main()
# }
