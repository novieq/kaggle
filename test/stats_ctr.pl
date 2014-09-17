#!/usr/bin/perl
#use strict;

my $usage = "$0 <train.csv> <out>";
my $fileName = shift @ARGV || die $usage;
my $outputFile = shift @ARGV || die $usage;

open(INFILE, "$fileName") || die "Cannot open $fileName";
my $line = <INFILE>;
#Id,Label,I1,I2,I3,I4,I5,I6,I7,I8,I9,I10,I11,I12,I13,C1,C2,C3,C4,C5,C6,C7,C8,C9,C10,C11,C12,C13,C14,C15,C16,C17,C18,C19,C20,C21,C22,C23,C24,C25,C26
my @aoh; #Array of hashes. Array location 0 is for the hash containing the categorical values of C1. 
my @aohClicks;

while($line = <INFILE>) {
	my @tokens = split(',', $line);
	foreach my $i (1..26) { #$i loops from C1 to C26 iterating over the respective hashes
		chomp $tokens[14+$i];
		if($tokens[14+$i]=="") { $tokens[14+$i]="**";}
		$aoh[$i]{$tokens[14+$i]}=$aoh[$i]{$tokens[14+$i]}+1; #Adding impressions for every categorical value of Ci
		$aohClicks[$i]{$tokens[14+$i]}=$aohClicks[$i]{$tokens[14+$i]}+$tokens[1];
	}
}
close(INFILE);

#foreach my $i (1..26) {
#	open(OUTPUT, ">C$i.$outputFile") || die "Cannot open $outputFile";
#	while (my ($k,$v)=each $aoh[$i]){print OUTPUT "C${i},$k,$v,$aohClicks[$i]{$k},",$aohClicks[$i]{$k}/$v,"\n"};
#	close OUTPUT;
#}

#open(OUTPUT, ">Summary") || die "Cannot open Summary";
#print OUTPUT "############# Summary ############\n";
#foreach my $i (1..26) {
#	print OUTPUT "C$i categories : ",scalar keys $aoh[$i],"\n";
#}
#close OUTPUT;

open(OUTFILE, ">train_ctr.csv") || die "Cannot open Summary";
open(INFILE, "train_random.100000.csv") || die "Cannot open $fileName";
$line = <INFILE>;

while($line = <INFILE>) {
        my @tokens = split(',', $line);
	my $outLine="";
	foreach my $i (0..14) {
		$outLine="$outLine$tokens[$i],";
	}
        foreach my $i (1..26) { #$i loops from C1 to C26 iterating over the respective hashes
                chomp $tokens[14+$i];
		if($tokens[14+$i]=="") { $tokens[14+$i]="**";}
		my $ctr;
		if($aoh[$i]{$tokens[14+$i]}>100) {
			$ctr = $aohClicks[$i]{$tokens[14+$i]}/$aoh[$i]{$tokens[14+$i]};
		} else {
			$ctr=0;
		}
                $outLine="$outLine$ctr,";
#		print "$i . $tokens[14+$i] $aohClicks[$i]{$tokens[14+$i]} $aoh[$i]{$tokens[14+$i]} $ctr $outLine\n";
        }
	print OUTFILE $outLine,"\n";
}
close INFILE;
close OUTFILE;

open(OUTFILE, ">test_cv_ctr.csv") || die "Cannot open Summary";
open(INFILE, "test_cv.csv") || die "Cannot open $fileName";
$line = <INFILE>;

while($line = <INFILE>) {
        my @tokens = split(',', $line);
        my $outLine="";
        foreach my $i (0..14) {
                $outLine="$outLine$tokens[$i],";
        }
        foreach my $i (1..26) { #$i loops from C1 to C26 iterating over the respective hashes
                chomp $tokens[14+$i];
                if($tokens[14+$i]=="") { $tokens[14+$i]="**";}
                my $ctr;
                if($aoh[$i]{$tokens[14+$i]}>100) {
                        $ctr = $aohClicks[$i]{$tokens[14+$i]}/$aoh[$i]{$tokens[14+$i]};
                } else {
                        $ctr=0;
                }
                $outLine="$outLine$ctr,";
#               print "$i . $tokens[14+$i] $aohClicks[$i]{$tokens[14+$i]} $aoh[$i]{$tokens[14+$i]} $ctr $outLine\n";
        }
        print OUTFILE $outLine,"\n";
}

close INFILE;
close OUTFILE;

open(OUTFILE, ">test_ctr.csv") || die "Cannot open Summary";
open(INFILE, "test.csv") || die "Cannot open $fileName";
$line = <INFILE>;

while($line = <INFILE>) {
        my @tokens = split(',', $line);
        my $outLine="";
        foreach my $i (0..13) {
                $outLine="$outLine$tokens[$i],";
        }
        foreach my $i (1..26) { #$i loops from C1 to C26 iterating over the respective hashes
                chomp $tokens[13+$i];
                if($tokens[13+$i]=="") { $tokens[13+$i]="**";}
                my $ctr;
                if($aoh[$i]{$tokens[13+$i]}>100) {
                        $ctr = $aohClicks[$i]{$tokens[13+$i]}/$aoh[$i]{$tokens[13+$i]};
                } else {
                        $ctr=0;
                }
                $outLine="$outLine$ctr,";
#               print "$i . $tokens[14+$i] $aohClicks[$i]{$tokens[14+$i]} $aoh[$i]{$tokens[14+$i]} $ctr $outLine\n";
        }
        print OUTFILE $outLine,"\n";
}

close INFILE;
close OUTFILE;
