#!/usr/bin/perl
#use strict;
my $usage = "$0 <train.csv>";
my $fileName = shift @ARGV || die $usage;
open(INFILE, "$fileName") || die "Cannot open $fileName";
my $line = <INFILE>;
while($line = <INFILE>) {
	my @varBuckets = split(',', $line);

}
